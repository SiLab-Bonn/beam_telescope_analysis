''' Implements the often needed split, map, combine paradigm '''
from __future__ import division

from multiprocessing import Pool, cpu_count, TimeoutError
from threading import Thread, Event
from collections import deque

import dill
import numpy as np
import tables as tb

from beam_telescope_analysis.tools import analysis_utils


def apply_async(pool, data, func, args=None, **kwargs):
    ''' Run fun(*args, **kwargs) in different process.

    fun can be a complex function since pickling is not done with the
    cpickle module as multiprocessing.apply_async would do, but with
    the more powerfull dill serialization.
    Additionally kwargs can be given and args can be given'''
    payload = dill.dumps((func, args, kwargs))
    return pool.apply_async(_run_with_dill, (data, payload,))


def _run_with_dill(data, payload):
    ''' Unpickle payload with dill.

    The payload is the function plus arguments and keyword arguments.
    '''
    func, args, kwargs = dill.loads(payload)
    if args:
        return func(data, *args, **(kwargs["func_kwargs"]))
    else:
        return func(data, **(kwargs["func_kwargs"]))


class SMC(object):
    def __init__(self, input_filename, output_filename, func, func_kwargs=None, node_desc=None, table=None, align_at=None, n_cores=None, mode='w', chunk_size=1000000):
        ''' Apply a function to a pytable on multiple cores in chunks.

            Parameters
            ----------
            input_filename : string
                Filename of the input file with the table.
            output_filename : string
                Filename of the output file with the resulting table/histogram.
                If the output file is the same as the input file, the filesize
                of the modified file is larger than it needs to be. This is due to the fact
                that HDF5 does not free space when removing nodes.
            func : function
                Worker function that is applied on the data from the input table.
            func_kwargs : dict
                Additional kwargs to pass to worker function
            node_desc : dict
                Output table/array parameters for pytables. Can be empty.
                Name/Filters/Title values are deduced from the input table
                if not defined. Data type is deduced from resulting data
                format if not defined.

                Example:
                node_desc = {'name': test}
                Would create an output node with the name test, the title and
                filters as the input table and the data type is deduced from
                the calculated data.
            table : string, list of strings, None
                If a string is given, use the node which has a name identical to the string. If the node does not exist, a RuntimeError is raised.
                If a list of strings is given, check for nodes that have a names identical to one of the strings from the list.
                The first existing node is used. Otherwise a RuntimeError is raised.
                If None, automatically search for a node. If multiple
                nodes exist in the input file, a RuntimeError is raised.
            align_at : string, None
                If specified align chunks at this column values
            n_cores : integer, None
                How many cores to use. If None, all available cores will be used.
                If set to 1, multithreading will be disabled which is useful for debuging.
            chunk_size : int
                Chunk size of the data when reading from file.

            Notes:
            ------
            It follows the split, map, combine paradigm:
            - split: data is splitted into chunks for multiple processes for
              speed increase
            - map: the function is called on each chunk. If the chunk per core
              is still too large to fit in memory it is chunked further. The
              result is written to a table per core.
            - combine: the tables are merged into one result table or one
              result histogram depending on the output data format
        '''
        # Set parameters
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.n_cores = n_cores
        self.align_at = align_at
        self.func = func
        node_desc = {} if node_desc is None else node_desc
        if not isinstance(node_desc, (list, tuple)):
            node_desc = (node_desc,)
        self.node_desc = node_desc
        self.node_name = None
        self.chunk_size = chunk_size
        self.mode = mode
        self.func_kwargs = {} if func_kwargs is None else func_kwargs

        if self.align_at is not None and self.align_at != 'event_number':
            raise NotImplementedError('Data alignment is only supported on event_number')

        # Get the table node name
        with tb.open_file(input_filename, mode='r') as in_file:
            if table is None:  # node is None, find a node
                tables = in_file.list_nodes('/', classname='Table')  # get all nodes of type 'table'
                if len(tables) == 1:  # if there is only one table, take this one
                    self.node_name = tables[0].name
                else:  # Multiple tables
                    raise RuntimeError('Paramter "table" not nto set and multiple table nodes found in file')
            elif isinstance(table, (list, tuple)):  # node is list of strings
                for node_name in table:
                    try:
                        in_file.get_node(in_file.root, node_name)
                        self.node_name = node_name
                        break  # stop at the first valid node
                    except tb.NoSuchNodeError:
                        pass
                if self.node_name is None:
                    raise RuntimeError('Found no table node with names: %s' % ', '.join(table))
            else:  # node is string
                try:
                    in_file.get_node(in_file.root, table)
                except tb.NoSuchNodeError:
                    raise RuntimeError('Found no table node with name: %s' % table)
                else:
                    self.node_name = table

            node = in_file.get_node(in_file.root, self.node_name)

            # Set number of rows
            self.n_rows = node.shape[0]

            for curr_node_desc in self.node_desc:
                # Set output parameters from input if not defined
                if 'name' not in curr_node_desc and len(self.node_desc) != 1:
                    raise ValueError('Key "name" must exist in "node_desc"')
                else:
                    curr_node_desc.setdefault('name', node.name)
                curr_node_desc.setdefault('filters', node.filters)

        # By default set n_cores to maximum cores available
        if not self.n_cores:
            self.n_cores = cpu_count()

        self.pool = None
        self.reader_thread = None
        self.async_worker_thread = None
        self.writer_thread = None
        self.out_file = None
        self.out_file_opened = Event()
        self.force_stop = Event()
        self.data_deque = deque()
        self.res_deque = deque()
        self.reader_thread = None
        # compute data
        self.split_map_combine()

    def _reader(self):
        if self.input_filename == self.output_filename:
            # wait for output file to be ready
            while not self.out_file_opened.wait(0.01) and not self.force_stop.wait(0.01):
                pass
            in_file = self.out_file
        else:
            try:
                in_file = tb.open_file(self.input_filename, mode='r')
            except Exception:
                self.force_stop.set()
                raise
        try:
            node = in_file.get_node(in_file.root, self.node_name)
        except Exception:
            self.force_stop.set()
            raise
        gen_data = analysis_utils.data_aligned_at_events(node, chunk_size=self.chunk_size)
        while not self.force_stop.wait(0.01):
            try:
                try:
                    data = next(gen_data)[0]
                except StopIteration:
                    break
            except Exception:
                self.force_stop.set()
                raise
            self.data_deque.append(data)
            while len(self.data_deque) >= 1 and not self.force_stop.wait(0.01):
                pass
        if self.input_filename != self.output_filename:
            try:
                in_file.close()
            except Exception:
                self.force_stop.set()
                raise
        self.data_deque.append(None)

    def _async_worker(self):
        while not self.force_stop.wait(0.01):
            try:
                data = self.data_deque.popleft()
            except IndexError:
                continue
            if data is None:
                break
            while len(self.res_deque) >= (self.n_cores + 1) and not self.force_stop.wait(0.01):
                pass
            res = apply_async(
                pool=self.pool,
                # fun=self._work,
                data=data,
                func=self.func,
                func_kwargs=self.func_kwargs)
            self.res_deque.append(res)
        self.res_deque.append(None)

    def _writer(self):
        init = True
        res = None
        try:
            self.out_file = tb.open_file(self.output_filename, mode=self.mode)
        except Exception:
            self.force_stop.set()
            raise
        self.out_file_opened.set()
        # Create result table later
        out_tables = []
        # Create result histogram later
        hists = []
        while not self.force_stop.wait(0.01):
            if res is None:
                try:
                    res = self.res_deque.popleft()
                except IndexError:
                    continue
                if res is None:
                    break
            try:
                res_data = res.get(timeout=0.01)
            except TimeoutError:
                continue
            res = None
            if not isinstance(res_data, (list, tuple)):
                res_data = (res_data,)
            # Loop over all tables and histograms
            for item in res_data:
                if not isinstance(item, np.ndarray):
                    raise RuntimeError('Return value of "func" must be numpy.ndarray')
            if not isinstance(self.node_desc, (list, tuple)):
                self.node_desc = (self.node_desc,)
            if len(res_data) != len(self.node_desc):
                raise RuntimeError('Return value of "func" does not match "node_desc"')
            # Create table or histogram on first iteration over data
            if init:
                for i, curr_node_desc in enumerate(self.node_desc):
                    if res_data[i].dtype.names:  # Recarray, thus table needed
                        # Create result table with specified data format
                        # If not provided, get description from returned data
                        if 'description' not in curr_node_desc:
                            # update dict
                            curr_node_desc['description'] = res_data[i].dtype
                        # create temporary node in case input and output node are the same
                        curr_node_desc["name"] = curr_node_desc["name"] + "_tmp"
                        table = self.out_file.create_table(
                            where=self.out_file.root,
                            **curr_node_desc)
                        out_tables.append(table)
                        hists.append(None)
                    else:  # Create histogram if data is not a Recarray
                        # Copy needed for reshape
                        out_tables.append(None)
                        hists.append(np.zeros_like(res_data[i]))
                init = False

            for i, curr_node_desc in enumerate(self.node_desc):
                if out_tables[i] is not None:
                    out_tables[i].append(res_data[i])  # Data is appended to the table
                    out_tables[i].flush()
                else:
                    # Check if array needs to be resized
                    new_shape = np.maximum(hists[i].shape, res_data[i].shape)
                    hists[i].resize(new_shape)
                    # Copym resize and add array to result
                    hist_copy = res_data[i].copy()
                    hist_copy.resize(new_shape)
                    hists[i] += hist_copy

        if not self.force_stop.is_set():
            for i, curr_node_desc in enumerate(self.node_desc):
                if res_data[i].dtype.names:  # Recarray
                    # rename temporary node
                    try:
                        self.out_file.remove_node(self.out_file.root, name=curr_node_desc["name"][:-4])
                    except tb.NoSuchNodeError:
                        pass
                    else:
                        self.out_file.flush()
                    self.out_file.rename_node(self.out_file.root, newname=curr_node_desc["name"][:-4], name=curr_node_desc["name"], overwrite=False)
                else:
                    # Store histogram to file
                    hist_dtype = hists[i].dtype
                    try:
                        self.out_file.remove_node(self.out_file.root, curr_node_desc["name"])
                    except tb.NoSuchNodeError:
                        pass
                    out_hist = self.out_file.create_carray(
                        where=self.out_file.root,
                        atom=tb.Atom.from_dtype(hist_dtype),
                        shape=hists[i].shape,
                        **curr_node_desc)
                    out_hist[:] = hists[i]
        try:
            self.out_file.close()
        except Exception:
            self.force_stop.set()
            raise

    def split_map_combine(self):
        self.pool = Pool(self.n_cores)

        self.out_file_opened.clear()
        self.force_stop.clear()
        self.writer_thread = Thread(target=self._writer, name='WriterThread')
        self.writer_thread.daemon = True
        self.writer_thread.start()

        self.async_worker_thread = Thread(target=self._async_worker, name='AsyncWorkerThread')
        self.async_worker_thread.daemon = True
        self.async_worker_thread.start()

        self.reader_thread = Thread(target=self._reader, name='ReaderThread')
        self.reader_thread.daemon = True
        self.reader_thread.start()

        self.reader_thread.join()
        self.reader_thread = None

        self.async_worker_thread.join()
        self.async_worker_thread = None

        self.writer_thread.join()
        self.writer_thread = None

        self.pool.close()
        self.pool.join()
        self.pool = None


if __name__ == '__main__':
    pass
