''' Minimal example script in order to show how the scattering plane feature for
    track fitting (only available when using the Kalman Filter) has to used.
'''

from beam_telescope_analysis.telescope.dut import ScatteringPlane


def run_analysis():
    # Create scattering planes and specifying needed parameters. All scattering planes will be added on the fly (during track fitting)
    # and will be sorted automatically according to their z-position.
    scattering_planes = [ScatteringPlane(name='ScatteringPlane1',
                                         material_budget=0.01,
                                         translation_x=0,
                                         translation_y=0,
                                         translation_z=1000.0,
                                         rotation_alpha=0,
                                         rotation_beta=0,
                                         rotation_gamma=0),
                         ScatteringPlane(name='ScatteringPlane2',
                                         material_budget=0.02,
                                         translation_x=0,
                                         translation_y=0,
                                         translation_z=2000.0,
                                         rotation_alpha=0,
                                         rotation_beta=0,
                                         rotation_gamma=0)]

    # In the track fitting step, `scattering_planes` needs just to be passed as a parameter to the track fitting fuction.


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows
    run_analysis()
