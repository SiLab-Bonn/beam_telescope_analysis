''' This module is a 'function sanctuary' of old  either not working functions or not really used
functions where no decicion is done yet to keep them, fx them or not.'''

# ALIGNMENT functions
# FIMXE: ALL FUNCTIONS BELOW NOT WORKING RIGHT NOW

from pykalman.standard import KalmanFilter


def plot_hit_alignment(title, difference, particles, ref_dut_column, table_column, actual_median, actual_mean, bins=100, output_pdf=None):
    plt.clf()
    plt.hist(difference, bins=bins, range=(-1. / 100. * np.amax(particles[:][ref_dut_column]) / 1., 1. / 100. * np.amax(particles[:][ref_dut_column]) / 1.))
    try:
        plt.yscale('log')
    except ValueError:
        pass
    plt.xlabel('%s - %s' % (ref_dut_column, table_column))
    plt.ylabel('#')
    plt.title(title)
    plt.grid()
    plt.plot([actual_median, actual_median], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Median %1.1f' % actual_median)
    plt.plot([actual_mean, actual_mean], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Mean %1.1f' % actual_mean)
    plt.legend(loc=0)
    if isinstance(output_pdf, PdfPages):
        output_pdf.savefig()
    elif output_pdf is True:
        plt.show()


def plot_hit_alignment_2(in_file_h5, combine_n_hits, median, mean, correlation, alignment, output_pdf=None):
    plt.clf()
    plt.xlabel('Hits')
    plt.ylabel('Offset')
    plt.grid()
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), median, linewidth=2.0, label='Median')
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), mean, linewidth=2.0, label='Mean')
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), correlation, linewidth=2.0, label='Alignment')
    plt.legend(loc=0)
    if isinstance(output_pdf, PdfPages):
        output_pdf.savefig()
    elif output_pdf is True:
        plt.show()


def plot_z(z, dut_z_col, dut_z_row, dut_z_col_pos_errors, dut_z_row_pos_errors, dut_index, output_pdf=None):
    plt.clf()
    plt.plot([dut_z_col.x, dut_z_col.x], [0., 1.], "--", label="DUT%d, col, z=%1.4f" % (dut_index, dut_z_col.x))
    plt.plot([dut_z_row.x, dut_z_row.x], [0., 1.], "--", label="DUT%d, row, z=%1.4f" % (dut_index, dut_z_row.x))
    plt.plot(z, dut_z_col_pos_errors / np.amax(dut_z_col_pos_errors), "-", label="DUT%d, column" % dut_index)
    plt.plot(z, dut_z_row_pos_errors / np.amax(dut_z_row_pos_errors), "-", label="DUT%d, row" % dut_index)
    plt.grid()
    plt.legend(loc=1)
    plt.ylim((np.amin(dut_z_col_pos_errors / np.amax(dut_z_col_pos_errors)), 1.))
    plt.xlabel('Relative z-position')
    plt.ylabel('Mean squared offset [a.u.]')
    plt.gca().set_yscale('log')
    plt.gca().get_yaxis().set_ticks([])
    if isinstance(output_pdf, PdfPages):
        output_pdf.savefig()
    elif output_pdf is True:
        plt.show()


def check_hit_alignment(input_tracklets_file, output_pdf_file, combine_n_hits=100000, correlated_only=False):
    '''Takes the tracklet array and plots the difference of column/row position of each DUT against the reference DUT0
    for every combine_n_events. If the alignment worked the median has to be around 0 and should not change with time
    (with the event number).

    Parameters
    ----------
    input_tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    output_pdf_file : pdf file name object
    combine_n_hits : int
        The number of events to combine for the hit position check
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('=== Check hit alignment ===')
    with tb.open_file(input_tracklets_file, mode="r") as in_file_h5:
        with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
            for table_column in in_file_h5.root.Tracklets.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    median, mean, std, alignment, correlation = [], [], [], [], []
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Check alignment for % s', table_column)
                    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Tracklets.shape[0], term_width=80)
                    progress_bar.start()
                    for index in range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits):
                        particles = in_file_h5.root.Tracklets[index:index + combine_n_hits]
                        particles = particles[np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)]  # only select events with hits in both DUTs
                        if correlated_only is True:
                            particles = particles[particles['track_quality'] & (1 << (24 + int(table_column[-1]))) == (1 << (24 + int(table_column[-1])))]
                        if particles.shape[0] == 0:
                            logging.warning('No correlation for dut %s and tracks %d - %d', table_column, index, index + combine_n_hits)
                            median.append(-1)
                            mean.append(-1)
                            std.append(-1)
                            alignment.append(0)
                            correlation.append(0)
                            continue
                        difference = particles[:][ref_dut_column] - particles[:][table_column]

                        # Calculate median, mean and RMS
                        actual_median, actual_mean, actual_rms = np.median(difference), np.mean(difference), np.std(difference)
                        alignment.append(np.median(np.abs(difference)))
                        correlation.append(difference.shape[0] * 100. / combine_n_hits)

                        median.append(actual_median)
                        mean.append(actual_mean)
                        std.append(actual_rms)

                        plot_hit_alignment('Aligned position difference for events %d - %d' % (index, index + combine_n_hits), difference, particles, ref_dut_column, table_column, actual_median, actual_mean, output_pdf, bins=64)
                        progress_bar.update(index)
                    plot_hit_alignment_2(in_file_h5, combine_n_hits, median, mean, correlation, alignment, output_pdf)
                    progress_bar.finish()


def fix_event_alignment(input_tracklets_file, tracklets_corr_file, input_alignment_file, error=3., n_bad_events=100, n_good_events=10, correlation_search_range=20000, good_events_search_range=100):
    '''Description

    Parameters
    ----------
    input_tracklets_file: pytables file
        Input file with original Tracklet data
    tracklets_corr_file: pyables_file
        Output file for corrected Tracklet data
    input_alignment_file: pytables file
        File with alignment data (used to get alignment fit errors)
    error: float
        Defines how much deviation between reference and observed DUT hit is allowed
    n_bad_events: int
        Detect no correlation when n_bad_events straight are not correlated
    n_good_events: int
    good_events_search_range: int
        n_good_events out of good_events_search_range must be correlated to detect correlation
    correlation_search_range: int
        Number of events that get checked for correlation when no correlation is found
    '''

    # Get alignment errors
    with tb.open_file(input_alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        n_duts = int(correlations.shape[0] / 2 + 1)
        column_sigma = np.zeros(shape=n_duts)
        row_sigma = np.zeros(shape=n_duts)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, n_duts):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    logging.info('=== Fix event alignment ===')

    with tb.open_file(input_tracklets_file, mode="r") as in_file_h5:
        particles = in_file_h5.root.Tracklets[:]
        event_numbers = np.ascontiguousarray(particles['event_number'])
        ref_column = np.ascontiguousarray(particles['column_dut_0'])
        ref_row = np.ascontiguousarray(particles['row_dut_0'])
        ref_charge = np.ascontiguousarray(particles['charge_dut_0'])

        particles_corrected = np.zeros_like(particles)

        particles_corrected['track_quality'] = (1 << 24)  # DUT0 is always correlated with itself

        for table_column in in_file_h5.root.Tracklets.dtype.names:
            if 'column_dut' in table_column and 'dut_0' not in table_column:
                column = np.ascontiguousarray(particles[table_column])  # create arrays for event alignment fixing
                row = np.ascontiguousarray(particles['row_dut_' + table_column[-1]])
                charge = np.ascontiguousarray(particles['charge_dut_' + table_column[-1]])

                logging.info('Fix alignment for % s', table_column)
                correlated, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, error=error, n_bad_events=n_bad_events, n_good_events=n_good_events, correlation_search_range=correlation_search_range, good_events_search_range=good_events_search_range)
                logging.info('Corrected %d places in the data', n_fixes)
                particles_corrected['event_number'] = event_numbers  # create new particles array with corrected values
                particles_corrected['column_dut_0'] = ref_column  # copy values that have not been changed
                particles_corrected['row_dut_0'] = ref_row
                particles_corrected['charge_dut_0'] = ref_charge
                particles_corrected['n_tracks'] = particles['n_tracks']
                particles_corrected[table_column] = column  # fill array with corrected values
                particles_corrected['row_dut_' + table_column[-1]] = row
                particles_corrected['charge_dut_' + table_column[-1]] = charge

                correlation_index = np.where(correlated == 1)[0]

                # Set correlation flag in track_quality field
                particles_corrected['track_quality'][correlation_index] |= (1 << (24 + int(table_column[-1])))

        # Create output file
        with tb.open_file(tracklets_corr_file, mode="w") as out_file_h5:
            try:
                out_file_h5.root.Tracklets._f_remove(recursive=True, force=False)
                logging.warning('Overwrite old corrected Tracklets file')
            except tb.NodeError:
                logging.info('Create new corrected Tracklets file')

            correction_out = out_file_h5.create_table(out_file_h5.root, name='Tracklets', description=in_file_h5.root.Tracklets.description, title='Corrected Tracklets data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            correction_out.append(particles_corrected)


def align_z(input_track_candidates_file, input_alignment_file, output_pdf_file, z_positions=None, track_quality=1, max_tracks=3, warn_at=0.5):
    '''Minimizes the squared distance between track hit and measured hit by changing the z position.
    In a perfect measurement the function should be minimal at the real DUT position. The tracks is given
    by the first and last reference hit. A track quality cut is applied to all cuts first.

    Parameters
    ----------
    input_track_candidates_file : pytables file
    input_alignment_file : pytables file
    output_pdf_file : pdf file name object
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
    '''
    logging.info('=== Find relative z-position ===')

    def pos_error(z, dut, first_reference, last_reference):
        return np.mean(np.square(z * (last_reference - first_reference) + first_reference - dut))

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
            track_candidates = in_file_h5.root.TrackCandidates[::10]  # take only every 10th track

            results = np.zeros((n_duts - 2,), dtype=[('DUT', np.uint8), ('z_position_column', np.float32), ('z_position_row', np.float32)])

            for dut_index in range(1, n_duts - 1):
                logging.info('Find best z-position for DUT %d', dut_index)
                dut_selection = (1 << (n_duts - 1)) | 1 | ((1 << (n_duts - 1)) >> dut_index)
                good_track_selection = np.logical_and((track_candidates['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8)), track_candidates['n_tracks'] <= max_tracks)
                good_track_candidates = track_candidates[good_track_selection]

                first_reference_row, last_reference_row = good_track_candidates['row_dut_0'], good_track_candidates['row_dut_%d' % (n_duts - 1)]
                first_reference_col, last_reference_col = good_track_candidates['column_dut_0'], good_track_candidates['column_dut_%d' % (n_duts - 1)]

                z = np.arange(0, 1., 0.01)
                dut_row = good_track_candidates['row_dut_%d' % dut_index]
                dut_col = good_track_candidates['column_dut_%d' % dut_index]
                dut_z_col = minimize_scalar(pos_error, args=(dut_col, first_reference_col, last_reference_col), bounds=(0., 1.), method='bounded')
                dut_z_row = minimize_scalar(pos_error, args=(dut_row, first_reference_row, last_reference_row), bounds=(0., 1.), method='bounded')
                dut_z_col_pos_errors, dut_z_row_pos_errors = [pos_error(i, dut_col, first_reference_col, last_reference_col) for i in z], [pos_error(i, dut_row, first_reference_row, last_reference_row) for i in z]
                results[dut_index - 1]['DUT'] = dut_index
                results[dut_index - 1]['z_position_column'] = dut_z_col.x
                results[dut_index - 1]['z_position_row'] = dut_z_row.x

                plot_z(z, dut_z_col, dut_z_row, dut_z_col_pos_errors, dut_z_row_pos_errors, dut_index, output_pdf)

    with tb.open_file(input_alignment_file, mode='r+') as out_file_h5:
        try:
            z_table_out = out_file_h5.createTable(out_file_h5.root, name='Zposition', description=results.dtype, title='Relative z positions of the DUTs without references', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            z_table_out.append(results)
        except tb.NodeError:
            logging.warning('Z position are do already exist. Do not overwrite.')

    z_positions_rec = np.add(([0.0] + results[:]['z_position_row'].tolist() + [1.0]), ([0.0] + results[:]['z_position_column'].tolist() + [1.0])) / 2.0

    if z_positions is not None:  # check reconstructed z against measured z
        z_positions_rec_abs = [i * z_positions[-1] for i in z_positions_rec]
        z_differences = [abs(i - j) for i, j in zip(z_positions, z_positions_rec_abs)]
        failing_duts = [j for (i, j) in zip(z_differences, range(5)) if i >= warn_at]
        logging.info('Absolute reconstructed z-positions %s', str(z_positions_rec_abs))
        if failing_duts:
            logging.warning('The reconstructed z positions are more than %1.1f cm off for DUTS %s', warn_at, str(failing_duts))
        else:
            logging.info('Difference between measured and reconstructed z-positions %s', str(z_differences))

    return z_positions_rec_abs if z_positions is not None else z_positions_rec

# TRACK ANALYSIS functions
def optimize_track_alignment(input_track_candidates_file, input_alignment_file, fraction=1, correlated_only=False):
    '''This step should not be needed but alignment checks showed an offset between the hit positions after alignment
    especially for DUTs that have a flipped orientation. This function corrects for the offset (c0 in the alignment).
    Does the same as optimize_hit_alignment but works on TrackCandidates file.
    If optimize_track_aligment is used track quality can change and must be calculated again from corrected data (use find_tracks_corr).


    Parameters
    ----------
    input_track_candidates_file : string
        Input file name with merged cluster hit table from all DUTs
    input_alignment_file : string
        Input file with alignment data
    use_fraction : float
        Use only every fraction-th hit for the alignment correction. For speed up. 1 means all hits are used
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('=== Optimize track alignment ===')
    with tb.open_file(input_track_candidates_file, mode="r+") as in_file_h5:
        particles = in_file_h5.root.TrackCandidates[:]
        with tb.open_file(input_alignment_file, 'r+') as alignment_file_h5:
            alignment_data = alignment_file_h5.root.Alignment[:]
            n_duts = alignment_data.shape[0] / 2
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    actual_dut = int(re.findall(r'\d+', table_column)[-1])
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Optimize alignment for % s', table_column)
                    particle_selection = particles[::fraction][np.logical_and(particles[::fraction][ref_dut_column] > 0, particles[::fraction][table_column] > 0)]  # only select events with hits in both DUTs
                    difference = particle_selection[ref_dut_column] - particle_selection[table_column]
                    selection = np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)  # select all hits from events with hits in both DUTs
                    particles[table_column][selection] += np.median(difference)
                    # Change linear offset of alignmet
                    if 'col' in table_column:
                        alignment_data['c0'][actual_dut - 1] -= np.median(difference)
                    else:
                        alignment_data['c0'][actual_dut + n_duts - 1] -= np.median(difference)
            # Store corrected/new alignment table after deleting old table
            alignment_file_h5.removeNode(alignment_file_h5.root, 'Alignment')
            result_table = alignment_file_h5.create_table(alignment_file_h5.root, name='Alignment', description=alignment_data.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            result_table.append(alignment_data)
        in_file_h5.removeNode(in_file_h5.root, 'TrackCandidates')
        corrected_trackcandidates_table = in_file_h5.create_table(in_file_h5.root, name='TrackCandidates', description=particles.dtype, title='TrackCandidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        corrected_trackcandidates_table.append(particles)


def check_track_alignment(trackcandidates_files, output_pdf_file, combine_n_hits=1000000, correlated_only=False, track_quality=None):
    '''Takes the tracklet array and plots the difference of column/row position of each DUT against the reference DUT0
    for every combine_n_events. If the alignment worked the median has to be around 0 and should not change with time
    (with the event number).
    Does the same as check_hit_alignment but works on TrackCandidates file.

    Parameters
    ----------
    input_track_candidates_file : string
        Input file name with merged cluster hit table from all DUTs
    output_pdf_file : pdf file name object
    combine_n_hits : int
        The number of events to combine for the hit position check
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    '''
    logging.info('=== Check TrackCandidates Alignment ===')
    with tb.open_file(trackcandidates_files, mode="r") as in_file_h5:
        with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    dut_index = int(table_column[-1])  # DUT index of actual DUT data
                    median, mean, std, alignment, correlation = [], [], [], [], []
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Check alignment for % s', table_column)
                    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.TrackCandidates.shape[0], term_width=80)
                    progress_bar.start()
                    for index in range(0, in_file_h5.root.TrackCandidates.shape[0], combine_n_hits):
                        particles = in_file_h5.root.TrackCandidates[index:index + combine_n_hits:5]  # take every 10th hit
                        particles = particles[np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)]  # only select events with hits in both DUTs
                        if correlated_only is True:
                            particles = particles[particles['track_quality'] & (1 << (24 + dut_index)) == (1 << (24 + dut_index))]
                        if track_quality:
                            particles = particles[particles['track_quality'] & (1 << (track_quality * 8 + dut_index)) == (1 << (track_quality * 8 + dut_index))]
                        if particles.shape[0] == 0:
                            logging.warning('No correlation for dut %s and events %d - %d', table_column, index, index + combine_n_hits)
                            median.append(-1)
                            mean.append(-1)
                            std.append(-1)
                            alignment.append(0)
                            correlation.append(0)
                            continue
                        difference = particles[:][ref_dut_column] - particles[:][table_column]

                        actual_median, actual_mean = np.median(difference), np.mean(difference)
                        alignment.append(np.median(np.abs(difference)))
                        correlation.append(difference.shape[0] * 100. / combine_n_hits)
                        plot_hit_alignment('Aligned position difference', difference, particles, ref_dut_column, table_column, actual_median, actual_mean, output_pdf, bins=100)

                        progress_bar.update(index)

                    progress_bar.finish()

def fit_tracks_kalman(input_track_candidates_file, output_tracks_file, geometry_file, z_positions, fit_duts=None, ignore_duts=None, include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], track_quality=1, max_tracks=None, output_pdf_file=None, use_correlated=False, method="Interpolation", pixel_size=[], chunk_size=1000000):
    '''Fits a line through selected DUT hits for selected DUTs. The selection criterion for the track candidates to fit is the track quality and the maximum number of hits per event.
    The fit is done for specified DUTs only (fit_duts). This DUT is then not included in the fit (include_duts). Bad DUTs can be always ignored in the fit (ignore_duts).

    Parameters
    ----------
    input_track_candidates_file : string
        file name with the track candidates table
    output_tracks_file : string
        file name of the created track file having the track table
    z_position : iterable
        the positions of the devices in z in cm
    fit_duts : iterable
        the duts to fit tracks for. If None all duts are used
    ignore_duts : iterable
        the duts that are not taken in a fit. Needed to exclude bad planes from track fit. Also included Duts are ignored!
    include_duts : iterable
        the relative dut positions of dut to use in the track fit. The position is relative to the actual dut the tracks are fitted for
        e.g. actual track fit dut = 2, include_duts = [-3, -2, -1, 1] means that duts 0, 1, 3 are used for the track fit
    max_tracks : int, None
        only events with tracks <= max tracks are taken
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    pixel_size : iterable, (x dimensions, y dimension)
        the size in um of the pixels, needed for chi2 calculation
    output_pdf_file : pdf file name object
        if None plots are printed to screen
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    method: string
        Defines the method for hit prediction:
            "Interpolation": chi2 minimization with straight line
            "Kalman": Kalman filter
    geometry_file: the file containing the geometry parameters (relative translation and angles)
    '''

    logging.info('=== Fit tracks ===')

    def create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts):
        # Define description
        description = [('event_number', np.int64)]
        for index in range(n_duts):
            description.append(('column_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('row_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('charge_dut_%d' % index, np.float))
        for dimension in range(3):
            description.append(('offset_%d' % dimension, np.float))
        for dimension in range(3):
            description.append(('slope_%d' % dimension, np.float))
        for index in range(n_duts):
            description.append(('predicted_x%d' % index, np.float))
        for index in range(n_duts):
            description.append(('predicted_y%d' % index, np.float))
        description.extend([('track_chi2', np.uint32), ('track_quality', np.uint32), ('n_tracks', np.uint8)])

        # Define structure of track_array
        tracks_array = np.zeros((n_tracks,), dtype=description)
        tracks_array['event_number'] = good_track_candidates['event_number']
        tracks_array['track_quality'] = good_track_candidates['track_quality']
        tracks_array['n_tracks'] = good_track_candidates['n_tracks']
        for index in range(n_duts):
            tracks_array['column_dut_%d' % index] = good_track_candidates['column_dut_%d' % index]
            tracks_array['row_dut_%d' % index] = good_track_candidates['row_dut_%d' % index]
            tracks_array['charge_dut_%d' % index] = good_track_candidates['charge_dut_%d' % index]
            intersection = offsets + slopes / slopes[:, 2, np.newaxis] * (z_positions[index] - offsets[:, 2, np.newaxis])  # intersection track with DUT plane
            tracks_array['predicted_x%d' % index] = intersection[:, 0]
            tracks_array['predicted_y%d' % index] = intersection[:, 1]
        for dimension in range(3):
            tracks_array['offset_%d' % dimension] = offsets[:, dimension]
            tracks_array['slope_%d' % dimension] = slopes[:, dimension]
        tracks_array['track_chi2'] = chi2s

        return tracks_array

    def create_results_array_kalman(good_track_candidates, track_estimates, chi2s, n_duts):
        # Define description
        description = [('event_number', np.int64)]
        for index in range(n_duts):
            description.append(('column_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('row_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('charge_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('predicted_x%d' % index, np.float))
        for index in range(n_duts):
            description.append(('predicted_y%d' % index, np.float))
        description.extend([('track_chi2', np.uint32), ('track_quality', np.uint32), ('n_tracks', np.uint8)])

        # Define structure of track_array
        tracks_array = np.zeros((n_tracks,), dtype=description)
        tracks_array['event_number'] = good_track_candidates['event_number']
        tracks_array['track_quality'] = good_track_candidates['track_quality']
        tracks_array['n_tracks'] = good_track_candidates['n_tracks']
        for index in range(n_duts):
            tracks_array['column_dut_%d' % index] = good_track_candidates['column_dut_%d' % index]
            tracks_array['row_dut_%d' % index] = good_track_candidates['row_dut_%d' % index]
            tracks_array['charge_dut_%d' % index] = good_track_candidates['charge_dut_%d' % index]
            tracks_array['predicted_x%d' % index] = track_estimates[:, index, 0]
            tracks_array['predicted_y%d' % index] = track_estimates[:, index, 1]
        tracks_array['track_chi2'] = chi2s

        return tracks_array

    method = method.lower()
    if method != "interpolation" and method != "kalman":
        raise ValueError('Method "%s" not recognized!' % method)
    if method == "kalman" and not pixel_size:
        raise ValueError('Kalman filter requires to provide pixel size for error measurement matrix covariance!')

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            with tb.open_file(output_tracks_file, mode='w') as out_file_h5:
                n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
                fit_duts = fit_duts if fit_duts else range(n_duts)
                for fit_dut in fit_duts:  # Loop over the DUTs where tracks shall be fitted for
                    logging.info('Fit tracks for DUT %d', fit_dut)
                    tracklets_table = None
                    for track_candidates_chunk, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates, chunk_size=chunk_size):

                        # Select track candidates
                        dut_selection = 0  # DUTs to be used in the fit
                        quality_mask = 0  # Masks DUTs to check track quality for
                        for include_dut in include_duts:  # Calculate mask to select DUT hits for fitting
                            if fit_dut + include_dut < 0 or ((ignore_duts and fit_dut + include_dut in ignore_duts) or fit_dut + include_dut >= n_duts):
                                continue
                            if include_dut >= 0:
                                dut_selection |= ((1 << fit_dut) << include_dut)
                            else:
                                dut_selection |= ((1 << fit_dut) >> abs(include_dut))

                            quality_mask = dut_selection | (1 << fit_dut)  # Include the DUT where the track is fitted for in quality check

                        if bin(dut_selection).count("1") < 2:
                            logging.warning('Insufficient track hits to do fit (< 2). Omit DUT %d', fit_dut)
                            continue

                        # Select tracks based on given track_quality
                        good_track_selection = (track_candidates_chunk['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8))
                        if max_tracks:  # Option to neglect events with too many hits
                            good_track_selection = np.logical_and(good_track_selection, track_candidates_chunk['n_tracks'] <= max_tracks)

                        logging.info('Lost %d tracks due to track quality cuts, %d percent ', good_track_selection.shape[0] - np.count_nonzero(good_track_selection), (1. - float(np.count_nonzero(good_track_selection) / float(good_track_selection.shape[0]))) * 100.)

                        if use_correlated:  # Reduce track selection to correlated DUTs only
                            good_track_selection &= (track_candidates_chunk['track_quality'] & (quality_mask << 24) == (quality_mask << 24))
                            logging.info('Lost due to correlated cuts %d', good_track_selection.shape[0] - np.sum(track_candidates_chunk['track_quality'] & (quality_mask << 24) == (quality_mask << 24)))

                        good_track_candidates_chunk = track_candidates_chunk[good_track_selection]

                        # Prepare track hits array to be fitted
                        n_fit_duts = bin(dut_selection).count("1")
                        index, n_tracks = 0, good_track_candidates_chunk['event_number'].shape[0]  # Index of tmp track hits array

                        translations, rotations = geometry_utils.recontruct_geometry_from_file(geometry_file)

                        if method == "interpolation":
                            track_hits = np.zeros((n_tracks, n_fit_duts, 3))
                        elif method == "kalman":
                            track_hits = np.zeros((n_tracks, n_duts, 3))
                        for dut_index in range(0, n_duts):  # Fill index loop of new array
                            if method == "interpolation" and (1 << dut_index) & dut_selection == (1 << dut_index):  # True if DUT is used in fit
                                xr = good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 0, 0] + good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 0, 1] + translations[dut_index, 0]
                                yr = good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 1, 1] + good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 1, 0] + translations[dut_index, 1]
                                xyz = np.column_stack((xr, yr, np.repeat(z_positions[dut_index], n_tracks)))
                                track_hits[:, index, :] = xyz
                                index += 1
                            elif method == "kalman":
                                if (1 << dut_index) & dut_selection == (1 << dut_index):  # TOCHECK! Not used = masked, OK, but also DUT must be masked...
                                    # xyz = np.column_stack(np.ma.array((good_track_candidates_chunk['column_dut_%s' % dut_index], good_track_candidates_chunk['row_dut_%s' % dut_index], np.repeat(z_positions[dut_index], n_tracks))))
                                    xr = good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 0, 0] + good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 0, 1] + translations[dut_index, 0]
                                    yr = good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 1, 1] + good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 1, 0] + translations[dut_index, 1]
                                    xyz = np.column_stack(np.ma.array((xr, yr, np.repeat(z_positions[dut_index], n_tracks))))
                                else:
                                    xr = good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 0, 0] + good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 0, 1] + translations[dut_index, 0]
                                    yr = good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 1, 1] + good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 1, 0] + translations[dut_index, 1]
                                    xyz = np.column_stack(np.ma.array((xr, yr, np.repeat(z_positions[dut_index], n_tracks)), mask=np.ones((n_tracks, 3))))
                                track_hits[:, index, :] = xyz
                                index += 1

                        # Split data and fit on all available cores
                        n_slices = cpu_count()
                        slice_length = np.ceil(1. * n_tracks / n_slices).astype(np.int32)

                        pool = Pool(n_slices)
                        if method == "interpolation":
                            slices = np.array_split(track_hits, n_slices)
                            results = pool.map(_fit_tracks_loop, slices)
                        elif method == "kalman":
                            slices = np.array_split(track_hits, n_slices)
                            slices = [(slice, pixel_size, z_positions) for slice in slices]
                            results = pool.map(_function_wrapper_fit_tracks_kalman_loop, slices)
                        pool.close()
                        pool.join()

                        # Store results
                        if method == "interpolation":
                            offsets = np.concatenate([i[0] for i in results])  # merge offsets from all cores in results
                            slopes = np.concatenate([i[1] for i in results])  # merge slopes from all cores in results
                            chi2s = np.concatenate([i[2] for i in results])  # merge chi2 from all cores in results
                            tracks_array = create_results_array(good_track_candidates_chunk, slopes, offsets, chi2s, n_duts)
                            if not tracklets_table:
                                tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                            tracklets_table.append(tracks_array)
                            for i in range(2):
                                mean, rms = np.mean(slopes[:, i]), np.std(slopes[:, i])
                                hist, edges = np.histogram(slopes[:, i], range=(mean - 5. * rms, mean + 5. * rms), bins=1000)
                                fit_ok = False
                                try:
                                    coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), mean, rms])
                                    fit_ok = True
                                except:
                                    fit_ok = False
                                plot_utils.plot_tracks_parameter(slopes, edges, i, hist, fit_ok, coeff, gauss, var_matrix, output_pdf, fit_dut, parName='Slope')
                                meano, rmso = np.mean(offsets[:, i]), np.std(offsets[:, i])
                                histo, edgeso = np.histogram(offsets[:, i], range=(meano - 5. * rmso, meano + 5. * rmso), bins=1000)
                                fit_ok = False
                                try:
                                    coeffo, var_matrixo = curve_fit(gauss, edgeso[:-1], histo, p0=[np.amax(histo), meano, rmso])
                                    fit_ok = True
                                except:
                                    fit_ok = False
                                plot_utils.plot_tracks_parameter(offsets, edgeso, i, histo, fit_ok, coeffo, gauss, var_matrixo, output_pdf, fit_dut, parName='Offset')
                        elif method == "kalman":
                            track_estimates = np.concatenate([i[0] for i in results])  # merge predicted x,y pos from all cores in results
                            chi2s = np.concatenate([i[1] for i in results])  # merge chi2 from all cores in results
                            tracks_array = create_results_array_kalman(good_track_candidates_chunk, track_estimates, chi2s, n_duts)
                            if not tracklets_table:
                                tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_Kalman_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks Kalman-smoothed for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                            tracklets_table.append(tracks_array)

                        # Plot chi2 distribution
                        plot_utils.plot_track_chi2(chi2s, fit_dut, output_pdf)

def _function_wrapper_fit_tracks_kalman_loop(args):  # Needed for multiprocessing call with arguments
    return _fit_tracks_kalman_loop(*args)


def _kalman_fit_3d(hits, transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance, observation_offset, initial_state_mean, initial_state_covariance, sigma):

    kf = KalmanFilter(
        transition_matrix, observation_matrix, transition_covariance, observation_covariance, transition_offset, observation_offset,
        initial_state_mean, initial_state_covariance
    )

    nplanes = hits.shape[0]
    meas = np.c_[hits, np.zeros((nplanes))]
    # kf = kf.em(meas, n_iter=5)
    smoothed_state_estimates = kf.smooth(meas)[0]

    chi2 = 0
    chi2 += np.sum(np.dot(np.square(meas[:, 0] - smoothed_state_estimates[:, 0]), np.square(1 / sigma)), dtype=np.double)

    return smoothed_state_estimates[:, 0], chi2


def _fit_tracks_kalman_loop(track_hits, pitches, plane_pos):
    nplanes = track_hits.shape[1]
    # TODO: from parameter
    thickness_si = [50. for _ in range(nplanes)]
    thickness_al = [600. for _ in range(nplanes)]
    thicknesses = thickness_si + thickness_al
    sigma = np.dot([pitches[i] for i in range(nplanes)], 1 / sqrt(12.))  # Resolution of each telescope plane
    # TOFIX
    tmp_plane_pos = [plane_pos[i] for i in range(nplanes)]
    # TOFIX : these two happens in case more planes are provided than dut files...

    ''' Calculations for multiple scattering'''
    X0si = 93600.  # radiation length in Silicon = 9.36 cm (Google Introduction to silicon trackers at LHC - TDX)
    X0al = 89000.  # in Aluminum
    energy = 100000.  # energy in MeV
    mass = 0.511  # mass in MeV
    momentum = sqrt(energy * energy - mass * mass)
    # beta = momentum / energy
    x0s = np.dot(thickness_si, 1 / X0si) + np.dot(thickness_al, 1 / X0al)
    thetas = np.zeros(nplanes, dtype=np.double)
    for i, xx in enumerate(x0s):
        thetat = ((13.6 / momentum) * sqrt(xx) * (1 + 0.038 * log(xx)))  # from formula
        thetas[i] = thetat.real
    print("Thetas: ")
    print(thetas)

    '''Kalman filter parameters'''
    transition_matrix = np.zeros((nplanes, 2, 2))
    for i, z in enumerate(plane_pos):
        transition_matrix[i] = [[1, 0], [0, 1]]
        if i < nplanes - 1:
            transition_matrix[i, 0, 1] = plane_pos[i + 1] - z
        else:
            transition_matrix[i, 0, 1] = transition_matrix[i - 1, 0, 1]
        if i >= nplanes - 1:
            break  # TOFIX

    transition_covariance = np.zeros((nplanes, 2, 2))
    for j, t in enumerate(thetas):
        transition_covariance[j] = [[t * t * thicknesses[j] * thicknesses[j] / 3, t * t * thicknesses[j] / 2], [t * t * thicknesses[j] / 2, t * t]]  # from some calculations

    transition_offset = [0, 0]
    # transition_covariance = [[theta * theta * thickness * thickness / 3, theta * theta * thickness / 2], [theta * theta * thickness / 2, theta * theta]]  # from some calculations
    observation_matrix = [[1, 0], [0, 0]]
    observation_offset = transition_offset
    observation_covariance_x = np.zeros((sigma.shape[0], 2, 2))
    observation_covariance_y = np.zeros((sigma.shape[0], 2, 2))
    observation_covariance_x[:, 0, 0] = np.square(sigma[:, 0])
    observation_covariance_y[:, 0, 0] = np.square(sigma[:, 1])

    ''' Initial state: first hit with slope 0, error: its sigma and a large one for slope '''
    initial_state_covariance_x = [[sigma[0, 0] ** 2, 0], [0, 0.01]]
    initial_state_covariance_y = [[sigma[0, 1] ** 2, 0], [0, 0.01]]

    track_estimates = np.zeros((track_hits.shape))

    chi2 = np.zeros((track_hits.shape[0]))

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        initial_state_mean_x = [actual_hits[0, 0], 0]
        initial_state_mean_y = [actual_hits[0, 1], 0]
        track_estimates_x, chi2x = _kalman_fit_3d(actual_hits[:, 0], transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance_x, observation_offset, initial_state_mean_x, initial_state_covariance_x, sigma[:, 0])
        track_estimates_y, chi2y = _kalman_fit_3d(actual_hits[:, 1], transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance_y, observation_offset, initial_state_mean_y, initial_state_covariance_y, sigma[:, 1])
        chi2[index] = chi2x + chi2y
        track_estimates[index, :, 0] = track_estimates_x
        track_estimates[index, :, 1] = track_estimates_y
        track_estimates[index, :, 2] = tmp_plane_pos

    return track_estimates, chi2
