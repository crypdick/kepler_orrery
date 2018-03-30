import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import datetime as dt
from diverging_map import diverge_map
import matplotlib.font_manager as fm
from matplotlib.ticker import FixedLocator as FL

# what KOI file to use
cd = os.path.abspath(os.path.dirname(__file__))
koilist = os.path.join(cd, 'KOI_List.txt')

# are we loading in system locations from a previous file (None if not)
centers_file = os.path.join(cd, 'orrery_centers.txt')
# lcenfile = None
# if we're not loading a centers file,
# where do we want to save the one generated (None if don't save)
# scenfile = os.path.join(cd, 'orrery_centers_2.txt')
scenfile = None

'''
# add in the solar system to the plots
addsolar = True
# put it at a fixed location? otherwise use posinlist to place it
fixedpos = False
# fixed x and y positions (in AU) to place the Solar System
# if addsolar and fixedpos are True
ssx = 6.
ssy = 1.
# fraction of the way through the planet list to treat the solar system
# if fixedpos is False.
# 0 puts it first and near the center, 1 puts it last on the outside
posinlist = 0.25
'''

# making rstart smaller or maxtry bigger takes longer but tightens the
# circle
# Radius of the circle (AU) to initially try placing a system
# when generating locations
min_radius = 4.
# number of tries to randomly place a system at a given radius
# before expanding the circle
max_placing_attempts = 20
# minimum spacing between systems (AU)
min_dist_between_systems = 0.3

# which font to use for the text
fontfile = os.path.join(cd, 'Avenir-Black.otf')
fontfam = 'normal'
fontcol = 'white'

# font sizes at various resolutions
fontsizes_1 = {480: 12, 720: 14, 1080: 22}
fontsizes_2 = {480: 15, 720: 17, 1080: 27}

# background color
background_color = 'black'

# color and alpha for the circular orbit paths
orbit_color = '#424242'
orbitalpha = 1.

# add a background to the legend to distinguish it?
legend_background = False
# if so, use this color and alpha
legend_bg_color = background_color
legend_alpha = 0.7

# are we making the png files for a movie or gif
makemovie = True
# resolution of the images. Currently support 480, 720 or 1080.
resolution = 480

# output directory for the images in the movie
# (will be created if it doesn't yet exist)
# outdir = os.path.join(cd, 'orrery-40s/')
outdir = os.path.join(cd, 'movie/')

# number of frames to produce
# using ffmpeg with the palette at (sec * frames/sec)
# nframes = 40 * 20
nframes = 2 * 30

# times to evaluate the planets at
# Kepler observed from 120.5 to 1591
times = np.arange(1591 - nframes / 2., 1591, 0.5)

# setup for the custom zoom levels
sys_indexed_by_size = np.arange(len(times))
nmax = sys_indexed_by_size[-1]
zooms = np.ones_like(times)

# what zoom level each frame is at (1. means default with everything)
"""
# zoom out once
zooms[inds < 0.25 * nmax] = 0.35
zooms[inds > 0.7 * nmax] = 1.
zooms[zooms < 0.] = np.interp(inds[zooms < 0.], inds[zooms > 0.],
                              zooms[zooms > 0.])
"""
"""
# zoom out then back in
zooms[inds < 0.25 * nmax] = 0.35
zooms[(inds > 0.5 * nmax) & (inds < 0.6 * nmax)] = 1.
zooms[inds > 0.85 * nmax] = 0.35
zooms[zooms < 0.] = np.interp(inds[zooms < 0.], inds[zooms > 0.],
                              zooms[zooms > 0.])
"""
# ===================================== #

# reference time for the Kepler data
time0 = dt.datetime(2009, 1, 1, 12)

# load in the data from the KOI list

star_ids, orbital_periods, dateof_1st_transit_detected, exo_radius, \
equilibrium_temperatures, orbit_semimajor_axis = np.genfromtxt(
    koilist, unpack=True, usecols=(1, 5, 8, 20, 26, 23), delimiter=',')


# grab the KICs with known parameters
good = (np.isfinite(orbit_semimajor_axis) & np.isfinite(orbital_periods) &
        np.isfinite(exo_radius) & np.isfinite(equilibrium_temperatures))

"""
# grab our solar sys
star_ids = star_ids[our_KIC_index]
orbital_periods = orbital_periods[our_KIC_index]
dateof_1st_transit_detected = dateof_1st_transit_detected[our_KIC_index]
orbit_semimajor_axis = orbit_semimajor_axis[our_KIC_index]
exo_radius = exo_radius[our_KIC_index]
planet_reach = exo_radius + orbit_semimajor_axis
equilibrium_temperatures = equilibrium_temperatures[our_KIC_index]
print(exo_radius)
"""

n_solar_systems = 1

"""
# the maximum size needed for each system
solarsys_furthest_reach = np.empty(n_solar_systems)
for ii in range(n_solar_systems):
    solarsys_furthest_reach[ii] = np.max(planet_reach[np.where(star_ids == uniq_solarsys_ids[ii])[0]])



for ii in range(n_solar_systems):
    # this seems to make a circle at the largest orbit
    c = plt.Circle((system_xcens[ii], system_ycens[ii]), solarsys_furthest_reach[ii], clip_on=False,
                   alpha=0.3)
    fig.gca().add_artist(c)
"""
system_xcens = np.array([0])
system_ycens = np.array([0])


# all of the parameters we need for the plot
t0s = dateof_1st_transit_detected
periods = orbital_periods
semis = orbit_semimajor_axis
radii = exo_radius
teqs = equilibrium_temperatures
used_planets = star_ids  # TODO if we throw out some systems update this
fullxcens = system_xcens
fullycens = system_ycens

# add our solar system
used_planets = np.zeros(8)
# always start the outer solar system in the same places
# for optimial visibility
t0s = np.array([85., 192., 266., 180.,
                            times[0] - 3. * 4332.8 / 4,
                            times[0] - 22. / 360 * 10755.7,
                            times[0] - 30687 * 145. / 360,
                            times[0] - 60190 * 202. / 360])
periods = np.array([87.97, 224.70, 365.26, 686.98,
                                    4332.8, 10755.7, 30687, 60190])
semis = np.array([0.387, 0.723, 1.0, 1.524, 5.203,
                                9.537, 19.19, 30.07])
radii = np.array([0.383, 0.95, 1.0, 0.53, 10.86, 9.00,
                                3.97, 3.86])
teqs = np.array([409, 299, 255, 206, 200,
                              200, 200, 200])
fullxcens = np.zeros(8)
fullycens = np.zeros(8)


if makemovie:
    plt.ioff()
else:
    plt.ion()

# create the figure at the right size (this assumes a default pix/inch of 100)
figsizes = {480: (8.54, 4.8), 720: (8.54, 4.8), 1080: (19.2, 10.8)}
fig = plt.figure(figsize=figsizes[resolution], frameon=False)

# make the plot cover the entire figure with the right background colors
ax = fig.add_axes([0.0, 0, 1, 1])
ax.axis('off')
fig.patch.set_facecolor(background_color)
plt.gca().patch.set_facecolor(background_color)

# don't count the orbits of the outer solar system in finding figure limits
#ns = np.where(used_planets != our_KIC_index)[0]

# this section manually makes the aspect ratio equal
#  but completely fills the figure

# need this much buffer zone so that planets don't get cut off
#buffsx = (fullxcens[ns].max() - fullxcens[ns].min()) * 0.007
#buffsy = (fullycens[ns].max() - fullycens[ns].min()) * 0.007
# current limits of the figure
#xmax = (fullxcens[ns] + semis[ns]).max() + buffsx
#xmin = (fullxcens[ns] - semis[ns]).min() - buffsx
#ymax = (fullycens[ns] + semis[ns]).max() + buffsy
#ymin = (fullycens[ns] - semis[ns]).min() - buffsy
#
# # figure aspect ratio
# sr = 16. / 9.
#
# # make the aspect ratio exactly right
# if (xmax - xmin) / (ymax - ymin) > sr:
#     plt.xlim(xmin, xmax)
#     plt.ylim((ymax + ymin) / 2. - (xmax - xmin) / (2. * sr),
#              (ymax + ymin) / 2. + (xmax - xmin) / (2. * sr))
# else:
#     plt.ylim(ymin, ymax)
#     plt.xlim((xmax + xmin) / 2. - (ymax - ymin) * sr / 2.,
#              (xmax + xmin) / 2. + (ymax - ymin) * sr / 2.)

lws = {480: 1, 720: 1, 1080: 2}
sslws = {480: 2, 720: 2, 1080: 4}

"""
# plot the orbital circles for every planet
for ii in range(n_planets):
    # solid, thinner lines for normal planets
    ls = 'solid'
    zo = 0
    lw = lws[resolution]
    # dashed, thicker ones for the solar system
    if used_planets[ii] == our_KIC_index:
        ls = 'dashed'
        zo = -3
        lw = sslws[resolution]

    c = plt.Circle((fullxcens[ii], fullycens[ii]), semis[ii], clip_on=False,
                   alpha=orbitalpha, fill=False,
                   color=orbit_color, zorder=zo, ls=ls, lw=lw)
    fig.gca().add_artist(c)
"""

# set up the planet size scale
sscales = {480: 12., 720: 30., 1080: 50.}
sscale = sscales[resolution]


radius_earth = 1.
rnep = 3.856
rjup = 10.864
rmerc = 0.383
# for the planet size legend
solarsys = np.array([rmerc, radius_earth, rnep, rjup])
pnames = ['Mercury', 'Earth', 'Neptune', 'Jupiter']
csolar = np.array([409, 255, 46, 112])

## keep the smallest planets visible and the largest from being too huge
#solarsys = np.clip(solarsys, 0.8, 1.3 * rjup)
solarscale = sscale * solarsys

#radii = np.clip(radii, 0.8, 1.3 * rjup)

pscale = sscale * radii

# color bar temperature tick values and labels
ticks = np.array([250, 500, 750, 1000, 1250])
labs = ['250', '500', '750', '1000', '1250', '1500']

# blue and red colors for the color bar
RGB1 = np.array([1, 185, 252])
RGB2 = np.array([220, 55, 19])

# create the diverging map with a white in the center
mycmap = diverge_map(RGB1=RGB1, RGB2=RGB2, numColors=15)

# just plot the planets at time 0. for this default plot
phase = 2. * np.pi * (0. - t0s) / periods
tmp = plt.scatter(fullxcens + semis * np.cos(phase),
                  fullycens + semis * np.sin(phase), marker='o',
                  edgecolors='none', lw=0, s=pscale, c=teqs, vmin=ticks.min(),
                  vmax=ticks.max(), zorder=3, cmap=mycmap, clip_on=False)

fsz1 = fontsizes_1[resolution]
fsz2 = fontsizes_2[resolution]
prop = fm.FontProperties(fname=fontfile)

"""
# create the 'Solar System' text identification
if addsolar:
    loc = np.where(usedkics == our_KIC_index)[0][0]
    plt.text(fullxcens[loc], fullycens[loc], 'Solar\nSystem', zorder=-2,
             color=fontcol, family=fontfam, fontproperties=prop, fontsize=fsz1,
             horizontalalignment='center', verticalalignment='center')
"""

"""
# if we're putting in a translucent background behind the text
# to make it easier to read
if legend_background:
    box1starts = {480: (0., 0.445), 720: (0., 0.46), 1080: (0., 0.47)}
    box1widths = {480: 0.19, 720: 0.147, 1080: 0.153}
    box1heights = {480: 0.555, 720: 0.54, 1080: 0.53}

    box2starts = {480: (0.79, 0.8), 720: (0.83, 0.84), 1080: (0.83, 0.84)}
    box2widths = {480: 0.21, 720: 0.17, 1080: 0.17}
    box2heights = {480: 0.2, 720: 0.16, 1080: 0.16}

    # create the rectangles at the right heights and widths
    # based on the resolution
    c = plt.Rectangle(box1starts[resolution], box1widths[resolution], box1heights[resolution],
                      alpha=legend_alpha, fc=legend_bg_color, ec='none', zorder=4,
                      transform=ax.transAxes)
    d = plt.Rectangle(box2starts[resolution], box2widths[resolution], box2heights[resolution],
                      alpha=legend_alpha, fc=legend_bg_color, ec='none', zorder=4,
                      transform=ax.transAxes)
    ax.add_artist(c)
    ax.add_artist(d)
"""

# appropriate spacing from the left edge for the color bar
cbxoffs = {480: 0.09, 720: 0.07, 1080: 0.074}
cbxoff = cbxoffs[resolution]

"""
# plot the solar system planet scale
ax.scatter(np.zeros(len(solarscale)) + cbxoff,
           1. - 0.13 + 0.03 * np.arange(len(solarscale)), s=solarscale,
           c=csolar, zorder=5, marker='o',
           edgecolors='none', lw=0, cmap=mycmap, vmin=ticks.min(),
           vmax=ticks.max(), clip_on=False, transform=ax.transAxes)

# put in the text labels for the solar system planet scale
for ii in np.arange(len(solarscale)):
    ax.text(cbxoff + 0.01, 1. - 0.14 + 0.03 * ii,
            pnames[ii], color=fontcol, family=fontfam,
            fontproperties=prop, fontsize=fsz1, zorder=5,
            transform=ax.transAxes)
"""


"""
# colorbar axis on the left centered with the planet scale
ax2 = fig.add_axes([cbxoff - 0.005, 0.54, 0.01, 0.3])
ax2.set_zorder(2)
cbar = plt.colorbar(tmp, cax=ax2, extend='both', ticks=ticks)
# remove the white/black outline around the color bar
cbar.outline.set_linewidth(0)
# allow two different tick scales
cbar.ax.minorticks_on()
# turn off tick lines and put the physical temperature scale on the left
cbar.ax.tick_params(axis='y', which='major', color=fontcol, width=2,
                    left='off', right='off', length=5, labelleft='on',
                    labelright='off', zorder=5)
# turn off tick lines and put the physical temperature approximations
# on the right
cbar.ax.tick_params(axis='y', which='minor', color=fontcol, width=2,
                    left='off', right='off', length=5, labelleft='off',
                    labelright='on', zorder=5)
# say where to put the physical temperature approximations and give them labels
cbar.ax.yaxis.set_minor_locator(FL(tmp.norm([255, 409, 730, 1200])))
cbar.ax.set_yticklabels(labs, color=fontcol, family=fontfam,
                        fontproperties=prop, fontsize=fsz1, zorder=5)
cbar.ax.set_yticklabels(['Earth', 'Mercury', 'Surface\nof Venus', 'Lava'],
                        minor=True, color=fontcol, family=fontfam,
                        fontproperties=prop, fontsize=fsz1)
clab = 'Planet Equilibrium\nTemperature (K)'
# add the overall label at the bottom of the color bar
cbar.ax.set_xlabel(clab, color=fontcol, family=fontfam, fontproperties=prop,
                   size=fsz1, zorder=5)
"""

# switch back to the main plot
plt.sca(ax)

# upper right credit and labels text offsets
txtxoffs = {480: 0.2, 720: 0.16, 1080: 0.16}
txtyoffs1 = {480: 0.10, 720: 0.08, 1080: 0.08}
txtyoffs2 = {480: 0.18, 720: 0.144, 1080: 0.144}

txtxoff = txtxoffs[resolution]
txtyoff1 = txtyoffs1[resolution]
txtyoff2 = txtyoffs2[resolution]

"""
# put in the credits in the top right
text = plt.text(1. - txtxoff, 1. - txtyoff1,
                time0.strftime('Kepler Orrery IV\n%d %b %Y'), color=fontcol,
                family=fontfam, fontproperties=prop,
                fontsize=fsz2, zorder=5, transform=ax.transAxes)
plt.text(1. - txtxoff, 1. - txtyoff2, 'By Ethan Kruse\n@ethan_kruse',
         color=fontcol, family=fontfam,
         fontproperties=prop, fontsize=fsz1,
         zorder=5, transform=ax.transAxes)
"""

# the center of the figure
x0 = np.mean(plt.xlim())
y0 = np.mean(plt.ylim())

# width of the figure
xdiff = np.diff(plt.xlim()) / 2.
ydiff = np.diff(plt.ylim()) / 2.

# create the output directory if necessary
if makemovie and not os.path.exists(outdir):
    os.mkdir(outdir)

if makemovie:
    # get rid of all old png files so they don't get included in a new movie
    oldfiles = glob(os.path.join(outdir, '*png'))
    for delfile in oldfiles:
        os.remove(delfile)

    # go through all the times and make the planets move
    for ii, time in enumerate(times):
        # remove old planet locations and dates
        tmp.remove()
        """
        text.remove()

        # re-zoom to appropriate level
        plt.xlim([x0 - xdiff * zooms[ii], x0 + xdiff * zooms[ii]])
        plt.ylim([y0 - ydiff * zooms[ii], y0 + ydiff * zooms[ii]])

        newt = time0 + dt.timedelta(time)
        # put in the credits in the top right
        text = plt.text(1. - txtxoff, 1. - txtyoff1,
                        newt.strftime('Kepler Orrery IV\n%d %b %Y'),
                        color=fontcol, family=fontfam,
                        fontproperties=prop,
                        fontsize=fsz2, zorder=5, transform=ax.transAxes)
        """
        # put the planets in the correct location
        phase = 2. * np.pi * (time - t0s) / periods
        print(phase)
        tmp = plt.scatter(fullxcens + semis * np.cos(phase),
                          fullycens + semis * np.sin(phase),
                          marker='o', edgecolors='none', lw=0, s=pscale, c=teqs,
                          vmin=ticks.min(), vmax=ticks.max(),
                          zorder=3, cmap=mycmap, clip_on=False)

        plt.savefig(os.path.join(outdir, 'fig{0:04d}.png'.format(ii)),
                    facecolor=fig.get_facecolor(), edgecolor='none')
        if not (ii % 10):
            print( '{0} of {1} frames'.format(ii, len(times)))

