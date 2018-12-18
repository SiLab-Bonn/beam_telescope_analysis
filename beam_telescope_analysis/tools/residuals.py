''' Script to simulate the effect of finite resolution (pixel pitch) in the residual plot.
A straight line fit through a DUT between two telescope planes is done.
For the 'fit' just the telescope planes are used and the fit is done in one dimension (x or y)
No multiple scattering and pixel charge sharing / cluster were considered. The result shows
the typical gauss in spikes structure. In reality the spikes are smeared by gaussians due to
multiple scattering.
'''
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_setup(telescope_pitch, DUT_pitch, z_positions, plane_offsets, beam_divergenz, beam_size, tracks_mc, tracks_reco, hits_mc, hits_track_reco_2, hits, plot_n_tracks=5):
    plt.clf()
    plt.xlim((z_positions[0] - 0.5, z_positions[-1] + 0.5))
    plt.ylim((-12 * telescope_pitch, 12 * telescope_pitch))
    plt.xlabel('z [cm]')
    plt.ylabel('x/y [$\mathrm{\mu}$m]')

    # Plot devices
    currentAxis = plt.gca()
    for index, z_position in enumerate(z_positions):
        if index == 1:
            for position in range(-10 * DUT_pitch, 10 * DUT_pitch, DUT_pitch):
                currentAxis.add_patch(Rectangle((z_position - 0.05, position + plane_offsets[index]), 0.1, DUT_pitch, facecolor="white"))
        else:
            for position in range(-10 * telescope_pitch, 10 * telescope_pitch, telescope_pitch):
                currentAxis.add_patch(Rectangle((z_position - 0.05, position + plane_offsets[index]), 0.1, telescope_pitch, facecolor="white"))

    # PLot some tracks with hits
    for i, track in enumerate(tracks_mc):
        plt.plot([z_positions[0], z_positions[2]], [track[0], track[1]], '-', linewidth=2, color='blue', label='MC track')
        plt.plot([z_positions[0], z_positions[2]], [tracks_reco[i][0], tracks_reco[i][1]], '--', linewidth=2, color='red', label='Reco track')
        for index, pixel in enumerate(hits[i]):  # Plot reco hit positions
            if index == 1:
                position = pixel * DUT_pitch + plane_offsets[index]
                currentAxis.add_patch(Rectangle((z_positions[index] - 0.05, position), 0.1, DUT_pitch, facecolor="Grey"))
            else:
                position = pixel * telescope_pitch + plane_offsets[index]
                currentAxis.add_patch(Rectangle((z_positions[index] - 0.05, position), 0.1, telescope_pitch, facecolor="Grey"))
        plt.plot([z_positions[1], z_positions[1]], [hits_mc[i], hits_mc[i]], 'o', linewidth=2, color='blue', label='MC hit')  # Plot mc hit positions
        plt.plot([z_positions[1], z_positions[1]], [hits_track_reco_2[i], hits_track_reco_2[i]], 'o', linewidth=2, color='red', label='Reco hit')  # Plot mc hit positions
        if i >= plot_n_tracks - 1:
            break
        if i == 0:
            plt.legend(loc=0)

    plt.show()


def plot_residuals(hits_reco_2, hits_track_reco_2, DUT_pitch):
    plt.clf()
    plt.grid()
    plt.hist(hits_reco_2 - hits_track_reco_2, bins=100, range=(-2 * DUT_pitch, 2 * DUT_pitch))
    plt.show()


if __name__ == '__main__':
    telescope_pitch = 50  # mu
    DUT_pitch = 50  # mu
    z_positions = [0., 20., 40.]  # cm
    plane_offsets = [0., 0., 0.]  # plane (DUT + telescope) offsets in mu, due to not perfect alignment, FIXME: domakes residual not centered

    n_tracks = 100000
    beam_divergenz = 1  # mrad
    beam_size = (0, 10)  # beam size in pixel

    np.random.seed(0)
    thetas = np.random.normal(0, beam_divergenz, n_tracks)
    beam_offsets = np.random.normal(0, 2.5 * DUT_pitch, n_tracks)

    tracks_mc = np.column_stack((beam_offsets, beam_offsets + (z_positions[-1] - z_positions[0]) * np.arctan(thetas * 0.001) * 1e4))  # in um, um, defined as x intersection on the telescope planes
    hits_mc_2 = tracks_mc[:, 0] + (tracks_mc[:, 1] - tracks_mc[:, 0]) / 2.  # real hit position in DUT

    # Digitized hit info (firing pixels) for all 3 planes
    hits_reco_1 = ((tracks_mc[:, 0] - plane_offsets[0]) / telescope_pitch).astype(np.int32)
    hits_reco_2 = ((tracks_mc[:, 0] + (tracks_mc[:, 1] - tracks_mc[:, 0]) / 2.) / DUT_pitch).astype(np.int32)
    hits_reco_3 = ((tracks_mc[:, 1] - plane_offsets[2]) / telescope_pitch).astype(np.int32)
    # Correct for int conversion that int(]-1 .. 1[) = 0
    hits_reco_1[tracks_mc[:, 0] - plane_offsets[0] < 0] = hits_reco_1[tracks_mc[:, 0] - plane_offsets[0] < 0] - 1
    hits_reco_2[tracks_mc[:, 0] + (tracks_mc[:, 1] - tracks_mc[:, 0]) / 2. < 0] = hits_reco_2[tracks_mc[:, 0] + (tracks_mc[:, 1] - tracks_mc[:, 0]) / 2. < 0] - 1
    hits_reco_3[tracks_mc[:, 1] - plane_offsets[2] < 0] = hits_reco_3[tracks_mc[:, 1] - plane_offsets[2] < 0] - 1
    hits_reco = np.column_stack((hits_reco_1, hits_reco_2, hits_reco_3))

    # Calculate track from telescope reco hits and the intersection with DUT plane
    tracks_reco = np.column_stack((hits_reco_1 * telescope_pitch + plane_offsets[0] + telescope_pitch / 2., hits_reco_3 * telescope_pitch + plane_offsets[2] + telescope_pitch / 2.))
    hits_track_reco_2 = tracks_reco[:, 0] + (tracks_reco[:, 1] - tracks_reco[:, 0]) / 2.
    # Calculate reco hit from firing pixel
    hits_pos_reco_2 = hits_reco_2 * DUT_pitch + plane_offsets[1] + DUT_pitch / 2.

    plot_setup(telescope_pitch, DUT_pitch, z_positions, plane_offsets, beam_divergenz, beam_size, tracks_mc, tracks_reco, hits_mc_2, hits_track_reco_2, hits_reco, plot_n_tracks=30)
    plot_residuals(hits_pos_reco_2, hits_track_reco_2, DUT_pitch)
