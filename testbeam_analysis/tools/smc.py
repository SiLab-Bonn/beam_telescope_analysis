''' Implements the often needed split, map, combine paradigm '''
from __future__ import division

import os
import tempfile
from multiprocessing import Pool, cpu_count

import dill
import numpy as np
import tables as tb


def apply_async(pool, fun, args=None, **kwargs):
    ''' Run fun(*args, **kwargs) in different process.

    fun can be a complex function since pickling is not done with the
    cpickle module as multiprocessing.apply_async would do, but with
    the more powerfull dill serialization.
    Additionally kwargs can be given and args can be given'''
    payload = dill.dumps((fun, args, kwargs))
    return pool.apply_async(_run_with_dill, (payload,))


def _run_with_dill(payload):
    ''' Unpickle payload with dill.

    The payload is the function plus arguments and keyword arguments.
    '''
    fun, args, kwargs = dill.loads(payload)
    if args:
        return fun(*args, **kwargs)
    else:
        return fun(**kwargs)


class SMC(object):

    def __init__(self, input_filename, output_filename,
                 func, func_kwargs={}, node_desc={}, table=None,
                 align_at=None, n_cores=None, mode='w', chunk_size=1000000):
        ''' Apply a function to a pytable on multiple cores in chunks.

            Parameters
            ----------
            input_filename : string
                Filename of the input file with the table.
            output_filename : string
                Filename of the output file with the resulting table/histogram.
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
        if not isinstance(node_desc, (list, tuple)):
            node_desc = (node_desc,)
        self.node_desc = node_desc
        self.node_name = None
        self.chunk_size = chunk_size
        self.mode = mode
        self.func_kwargs = func_kwargs

        if self.align_at and self.align_at != 'event_number':
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

        if not self.n_cores:  # Set n_cores to maximum cores available
            self.n_cores = cpu_count()
            # Deactivate multithreading for small data sets
            # Overhead of pools can make multiprocesssing slower
            if self.n_rows < 2. * self.chunk_size:
                self.n_cores = 1

        # The three main steps
        self._split()
        self._map()
        self._combine()

    def _split(self):
        self.start_i, self.stop_i = self._get_split_indeces()
        assert len(self.start_i) == len(self.stop_i)

    def _map(self):
        chunk_size_per_core = int(self.chunk_size / self.n_cores)
        if self.n_cores == 1:
            self.tmp_files = [self._work(input_filename=self.input_filename,
                                         node_name=self.node_name,
                                         func=self.func,
                                         func_kwargs=self.func_kwargs,
                                         node_desc=self.node_desc,
                                         start_i=self.start_i[0],
                                         stop_i=self.stop_i[0],
                                         chunk_size=chunk_size_per_core)]
        else:
            # Run function in parallel
            pool = Pool(self.n_cores)

            jobs = []
            for i in range(self.n_cores):
                job = apply_async(pool=pool,
                                  fun=self._work,
                                  input_filename=self.input_filename,
                                  node_name=self.node_name,
                                  func=self.func,
                                  func_kwargs=self.func_kwargs,
                                  node_desc=self.node_desc,
                                  start_i=self.start_i[i],
                                  stop_i=self.stop_i[i],
                                  chunk_size=chunk_size_per_core)
                jobs.append(job)

            # Gather results
            self.tmp_files = []
            for job in jobs:
                self.tmp_files.append(job.get())

            pool.close()
            pool.join()

            del pool

    def _work(self, input_filename, node_name, func, func_kwargs,
              node_desc, start_i, stop_i, chunk_size):
        ''' Defines the work per worker.

        Reads data, applies the function and stores data in chunks into a table
        or a histogram.
        '''
        with tb.open_file(input_filename, mode='r') as in_file:
            node = in_file.get_node(in_file.root, node_name)
            output_file = tempfile.NamedTemporaryFile(delete=False, dir=os.getcwd())
            with tb.open_file(output_file.name, mode='w') as out_file:
                # Create result table later
                table_out = []
                # Create result histogram later
                hist_out = []

                for data, _ in self._chunks_at_event(table=node,
                                                     start_index=start_i,
                                                     stop_index=stop_i,
                                                     chunk_size=chunk_size):

                    data_ret = func(data, **func_kwargs)
                    if not isinstance(data_ret, (list, tuple)):
                        data_ret = (data_ret,)
                    if not isinstance(node_desc, (list, tuple)):
                        node_desc = (node_desc,)
                    if len(data_ret) != len(node_desc):
                        raise RuntimeError('Return value of "func" does not match "node_desc"')
                    # Loop over all tables and histograms
                    for i, curr_node_desc in enumerate(node_desc):
                        if not isinstance(data_ret[i], np.ndarray):
                            raise RuntimeError('Return value of "func" must be numpy.ndarray')
                        # Create table or histogram on first iteration over data
                        if len(table_out) != len(node_desc):
                            if data_ret[i].dtype.names:  # Recarray, thus table needed
                                # Create result table with specified data format
                                # If not provided, get description from returned data
                                if 'description' not in curr_node_desc:
                                    # update dict
                                    curr_node_desc['description'] = data_ret[i].dtype
                                table_out.append(out_file.create_table(where=out_file.root,
                                                                       **curr_node_desc))
                                hist_out.append(None)
                            else:  # Create histogram if data is not a table
                                # Copy needed for reshape
                                table_out.append(None)
                                hist_out.append(np.zeros_like(data_ret[i]))

                        if table_out[i] is not None:
                            table_out[i].append(data_ret[i])  # Data is appended to the table
                        else:
                            # Check if array needs to be resized
                            new_shape = np.maximum(hist_out[i].shape, data_ret[i].shape)
                            hist_out[i].resize(new_shape)
                            # Add array
                            data_ret[i].resize(new_shape)
                            hist_out[i] += data_ret[i]

                for i, curr_node_desc in enumerate(node_desc):
                    # Create CArray for histogram
                    if hist_out and hist_out[i] is not None:
                        # Store histogram to file
                        hist_dtype = hist_out[i].dtype
                        out = out_file.create_carray(where=out_file.root,
                                                     atom=tb.Atom.from_dtype(hist_dtype),
                                                     shape=hist_out[i].shape,
                                                     **curr_node_desc)
                        out[:] = hist_out[i]

        return output_file.name

    def _combine(self):
        file_mode = self.mode
        for curr_node_desc in self.node_desc:
            curr_node_name = curr_node_desc['name']
            # Check data type to decide on combine procedure
            with tb.open_file(self.tmp_files[0], mode='r') as in_file:
                input_node = in_file.get_node(in_file.root, curr_node_name)
                if isinstance(input_node, tb.carray.CArray):
                    data_type = 'array'
                else:
                    data_type = 'table'

            if data_type == 'table':
                with tb.open_file(self.output_filename, mode=file_mode) as out_file:
                    for index, tmp_file in enumerate(self.tmp_files):
                        with tb.open_file(tmp_file, mode="r") as in_file:
                            input_node = in_file.get_node(in_file.root, curr_node_name)
                            # Copy node from first result file to output file
                            if index == 0:
                                try:
                                    out_file.remove_node(out_file.root, curr_node_name)
                                except tb.NoSuchNodeError:
                                    pass
                                out_file.copy_node(input_node, out_file.root, overwrite=False, recursive=True)
                                output_node = out_file.get_node(out_file.root, curr_node_name)
                            else:
                                output_node.append(input_node[:])
            else:
                # Merge arrays from several temprary files,
                # merge them by adding up array data and resizing shape if necessary
                with tb.open_file(self.output_filename, mode=file_mode) as out_file:
                    for index, tmp_file in enumerate(self.tmp_files):
                        with tb.open_file(tmp_file, mode='r') as in_file:
                            tmp_data = in_file.get_node(in_file.root, curr_node_name)[:]
                            if index == 0:
                                # Copy needed for reshape
                                hist_data = tmp_data.copy()
                            else:
                                # Check if array needs to be resized
                                new_shape = np.maximum(hist_data.shape, tmp_data.shape)
                                hist_data.resize(new_shape)
                                # Add array
                                tmp_data.resize(new_shape)
                                hist_data += tmp_data

                    data_dtype = hist_data.dtype
                    try:
                        out_file.remove_node(out_file.root, curr_node_name)
                    except tb.NoSuchNodeError:
                        pass
                    out = out_file.create_carray(where=out_file.root,
                                                 atom=tb.Atom.from_dtype(data_dtype),
                                                 shape=hist_data.shape,
                                                 **curr_node_desc)
                    out[:] = hist_data
            # append to output file if more than one node exists
            file_mode = 'r+'

        for tmp_file in self.tmp_files:
            os.remove(tmp_file)

    def _get_split_indeces(self):
        ''' Calculates the data range for each core.

            Return two lists with start/stop indeces.
            Stop indeces are exclusive.
        '''
        core_chunk_size = self.n_rows // self.n_cores
        start_indeces = list(range(0,
                                   self.n_rows,
                                   core_chunk_size)
                             [:self.n_cores])

        if not self.align_at:
            stop_indeces = start_indeces[1:]
        else:
            stop_indeces = self._get_next_index(start_indeces)
            start_indeces = [0] + stop_indeces

        stop_indeces.append(self.n_rows)  # Last index always table size

        assert len(stop_indeces) == self.n_cores
        assert len(start_indeces) == self.n_cores

        return start_indeces, stop_indeces

    def _get_next_index(self, indeces):
        ''' Get closest index where the alignment column changes '''

        next_indeces = []
        for index in indeces[1:]:
            with tb.open_file(self.input_filename, mode='r') as in_file:
                node = in_file.get_node(in_file.root, self.node_name)
                values = node[index:index + self.chunk_size][self.align_at]
                value = values[0]
                for i, v in enumerate(values):
                    if v != value:
                        next_indeces.append(index + i)
                        break
                    value = v

        return next_indeces

    def _chunks_at_event(self, table, start_index=None, stop_index=None,
                         chunk_size=1000000):
        '''Takes the table with a event_number column and returns chunks.

        The chunks are chosen in a way that the events are not splitted.
        Start and the stop indices limiting the table size can be specified to
        improve performance. The event_number column must be sorted.

        Parameters
        ----------
        table : pytables.table
            The data.
        start_index : int
            Start index of data. If None, no limit is set.
        stop_index : int
            Stop index of data. If None, no limit is set.
        chunk_size : int
            Maximum chunk size per read.

        Returns
        -------
        Iterator of tuples
            Data of the actual data chunk and start index for the next chunk.

        Example
        -------
        for data, index in chunk_aligned_at_events(table):
            do_something(data)
            show_progress(index)
        '''
        # Initialize variables
        if not start_index:
            start_index = 0
        if not stop_index:
            stop_index = table.shape[0]

        # Limit max index
        if stop_index > table.shape[0]:
            stop_index = table.shape[0]

        # Special case, one read is enough, data not bigger than one chunk and
        # the indices are known
        if start_index + chunk_size >= stop_index:
            yield table.read(start=start_index, stop=stop_index), stop_index
        else:  # Read data in chunks, chunks do not divide events
            current_start_index = start_index
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size,
                                         stop_index)
                chunk = table[current_start_index:current_stop_index]
                if current_stop_index == stop_index:  # Last chunk
                    yield chunk, stop_index
                    break

                # Find maximum non event number splitting index
                event_numbers = chunk["event_number"]
                last_event = event_numbers[-1]

                # Search for next event number
                chunk_stop_i = np.searchsorted(event_numbers,
                                               last_event,
                                               side="left")

                yield chunk[:chunk_stop_i], current_start_index + chunk_stop_i

                current_start_index += chunk_stop_i


if __name__ == '__main__':
    pass
