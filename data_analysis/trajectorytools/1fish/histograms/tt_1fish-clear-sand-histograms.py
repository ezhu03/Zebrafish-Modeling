"""
This script loads single-fish idtracker.ai trajectories for either Clear or Sanded tanks
(optionally blind fish), converts them to physical units, and produces:
  • Spatial heatmaps (positions and turning locations)
  • Histograms for r, speed, radial speed, and near-wall angle φ
  • Scatter plots of kinematic relationships

It validates user input, aggregates multiple sessions per age (7/14/21 dpf or 70/140/210),
computes derived radial/tangential components, estimates near-wall turning windows, and
summarizes mean turning time by age.

Adjustable parameters live in the CONFIG block below.
"""

# ---------- Imports ----------
from pprint import pprint
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import pandas as pd
import trajectorytools as tt
import matplotlib.colors as mcolors
import sys

# iterate through the dpf values

# ==========================
# CONFIGURATION (adjustable)
# ==========================
RADIUS = 5.0                 # cm, tank radius used for physical scaling and plotting bounds
NEAR_WALL_FRACTION = 0.8     # r/RADIUS threshold to consider a frame "near wall"
TURN_PHI_WINDOW = 0.4        # radians, initial φ window when entering near-wall band
BINS_2D = 10                 # 2D heatmap bins per axis
CIRCLE_POINTS = 300          # resolution for boundary circle drawing
FIG_DPI = 100                # on-screen figure DPI
SAVE_DPI = 3000              # exported PNG DPI
ANGLE_STEP = 0.01            # step for sampling boundary angles (radians)
COLORMAP_LIGHTB = "light:b"  # seaborn light blue colormap name (for 2D histograms)

# idtracker.ai camera geometry constants (px). 1048 is center; 2048 is diameter; 1024 = 2048/2
PX_CENTER = 1048
PX_DIAMETER = 2048
PX_HALF = 1024

Y_FLIP = -1                  # multiply y by this to align coordinate convention

# All runtime choices come from the user; data files are selected based on surface/age/vision.

# ---------- USER INPUT VALIDATION ----------
val_raw = input('Tank surface (Sanded/Clear): ').strip().lower()
if val_raw == 'sanded':
    arr = [70, 140, 210]
elif val_raw == 'clear':
    arr = [7, 14, 21]
else:
    print("Invalid input for tank surface. Please enter 'Sanded' or 'Clear'. Exiting.")
    sys.exit(1)

indiv_raw = input('Individual plots? (Y/N): ').strip().upper()
if indiv_raw not in ('Y', 'N'):
    print("Invalid input for 'Individual plots?'. Please enter 'Y' or 'N'. Exiting.")
    sys.exit(1)
indiv = indiv_raw

blind_raw = input('Blind fish? (Y/N): ').strip().upper()
if blind_raw not in ('Y', 'N'):
    print("Invalid input for 'Blind fish?'. Please enter 'Y' or 'N'. Exiting.")
    sys.exit(1)
blind = blind_raw

outputs = []
tt_avg = []
tt_std = []

# Use a local variable that mirrors the config constant (avoids editing function bodies)
radius = RADIUS

# ---------- Helpers ----------

def count_ones(array):
    count = 0
    for num in array:
        if num == 1:
            count += 1
    return count

def openfile(file, sigma=1):
    """Open a trajectory file as a tt.Trajectories (with smoothing)."""
    tr = tt.Trajectories.from_idtrackerai(
        file,
        interpolate_nans=True,
        smooth_params={'sigma': sigma}
    )
    return tr

def processtr(tr):
    """Center, rescale, and set time/length units. Prints diagnostic ranges."""
    center, r_est = tr.estimate_center_and_radius_from_locations(in_px=True)
    tr.origin_to(center)
    tr.new_length_unit(tr.params['body_length_px'], 'BL')
    tr.new_time_unit(tr.params['frame_rate'], 's')
    print('Positions:')
    print('X range:', np.nanmin(tr.s[..., 0]), np.nanmax(tr.s[..., 0]), 'BL')
    print('Y range:', np.nanmin(tr.s[..., 1]), np.nanmax(tr.s[..., 1]), 'BL')
    print('Velocities:')
    print('X range:', np.nanmin(tr.v[..., 0]), np.nanmax(tr.v[..., 0]), 'BL/s')
    print('Y range:', np.nanmin(tr.v[..., 1]), np.nanmax(tr.v[..., 1]), 'BL/s')
    print('Accelerations:')
    print('X range:', np.nanmin(tr.a[..., 0]), np.nanmax(tr.a[..., 0]), 'BL/s^2')
    print('Y range:', np.nanmin(tr.a[..., 1]), np.nanmax(tr.a[..., 1]), 'BL/s^2')
    pprint(tr.params)
    return tr

def plotReflection(xposition, yposition, xvelocity, yvelocity):
    """Estimate fraction of boundary points visible as reflection."""
    mag = np.sqrt(xposition ** 2 + yposition ** 2)
    magv = np.sqrt(xvelocity ** 2 + yvelocity ** 2)
    distance = 2 * (radius - mag)  # kept for clarity; not used downstream but documents geometry

    angles = np.arange(0, 2 * np.pi, ANGLE_STEP)
    xbound = radius * np.cos(angles)
    ybound = radius * np.sin(angles)
    labels = np.zeros(len(angles))

    # Thresholds (radians) for geometric constraints
    THETA_MIN = 0.85
    THETA_MAX = 2.29
    PHI_MAX = 2.958

    for i in range(len(angles)):
        magd = np.sqrt((xbound[i] - xposition) ** 2 + (ybound[i] - yposition) ** 2)
        theta = np.arccos((xbound[i] * (xbound[i] - xposition) + ybound[i] * (ybound[i] - yposition)) / (radius * magd))
        phi = np.arccos((xvelocity * (xbound[i] - xposition) + yvelocity * (ybound[i] - yposition)) / (magv * magd))
        if (theta > THETA_MIN) and (theta < THETA_MAX) and (phi < PHI_MAX):
            labels[i] = 1
    return count_ones(labels) / len(labels)

def border_turning(tr, refl_prop):
    """
    Convert positions/velocities to physical units with our radius convention and
    accumulate the reflection proportion per frame into `refl_prop` (list).
    """
    offset = tr.params['_center'] - np.array([PX_CENTER, PX_CENTER])

    pos1 = tr.s * tr.params['length_unit'] * (2 * radius / PX_DIAMETER) + offset * radius / PX_HALF
    pos1 = np.array(pos1.reshape(pos1.shape[0], 2))
    for i in range(len(pos1)):
        pos1[i][1] *= Y_FLIP

    v1 = np.array(tr.v).reshape(tr.v.shape[0], 2)
    for i in range(len(v1)):
        v1[i][1] *= Y_FLIP

    norms = np.linalg.norm(v1, axis=1)
    v1 = v1 / norms[:, np.newaxis]

    i = 0
    while i < len(pos1):
        prop = plotReflection(pos1[i][0], pos1[i][1], v1[i][0], v1[i][1])
        refl_prop.append(prop)
        i += 1


# ----- Main loop over requested ages (arr) -----
for x in arr:
    path_raw = input('Physical Path? (Y/N): ').strip().upper()
    if path_raw not in ('Y', 'N'):
        print("Invalid input for 'Physical Path?'. Please enter 'Y' or 'N'. Exiting.")
        sys.exit(1)
    path = path_raw

    # ----- File lists per age/surface/vision -----
    if x == 7:
        file1 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear1/trajectories/trajectories.npy"
        file2 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear2/trajectories/trajectories.npy"
        file3 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear3/trajectories/trajectories.npy"
        file4 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear4/trajectories/trajectories.npy"
        file5 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear5/trajectories/trajectories.npy"
        file6 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy"
        file7 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear2/trajectories/validated.npy"
        file8 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear3/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5, file6, file7, file8]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file8 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            files = [file1, file2, file3, file4, file5, file6, file7, file8]
    elif x == 14:
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear5/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            files = [file1, file2, file3, file5, file6, file7]
    elif x == 21:
        file1 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file2 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file3 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file4 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file5 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file6 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file7 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear4/trajectories/validated.npy"
        file8 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear5/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5, file6, file7, file8]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear3-crispr/trajectories/validated.npy"
            files = [file1, file2, file3]
    elif x == 70:
        file1 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded1/trajectories/validated.npy"
        file2 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded2/trajectories/validated.npy"
        file3 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded3/trajectories/validated.npy"
        file4 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded1/trajectories/trajectories.npy"
        file5 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded2/trajectories/trajectories.npy"
        file6 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded3/trajectories/trajectories.npy"
        file7 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded4/trajectories/trajectories.npy"
        file8 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded5/trajectories/trajectories.npy"
        files = [file1, file2, file3, file4, file5, file6, file7, file8]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            files = [file1, file2, file3, file5, file6, file7]
    elif x == 140:
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded5/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            files = [file1, file2, file3, file4, file5, file6, file7]
    elif x == 210:
        file1 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file2 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file3 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file4 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file5 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file6 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file7 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded4/trajectories/validated.npy"
        file8 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded5/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5, file6, file7, file8]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded3-crispr/trajectories/validated.npy"
            files = [file1, file2, file3]

    # ----- Load & process each provided file -----
    trs = []
    for file in files:
        tr_temp = openfile(file)
        trs.append(tr_temp)

    refl_prop = []
    processedpos = []
    processedvel = []

    for temp in trs:
        processed_temp = processtr(temp)
        border_turning(processed_temp, refl_prop)

        offset = processed_temp.params['_center'] - np.array([PX_CENTER, PX_CENTER])
        temppos = processed_temp.s * processed_temp.params['length_unit'] * (2 * radius / PX_DIAMETER) + offset * radius / PX_HALF
        tempvel = processed_temp.v * (processed_temp.params['length_unit'] / processed_temp.params['time_unit']) * (2 * radius / PX_DIAMETER)
        processedpos.append(temppos)

        if path == 'Y':
            # ----- Optional: draw physical path with color-graded arrows -----
            all_positions = np.reshape(np.array(temppos), (-1, 2))
            allxpos = all_positions[:, 0]
            allypos = all_positions[:, 1]

            fig, ax = plt.subplots(figsize=(7.5, 6))
            plt.rcParams['figure.dpi'] = FIG_DPI

            # Boundary circle
            theta = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
            xc = radius * np.cos(theta)
            yc = radius * np.sin(theta)
            ax.plot(xc, yc, label=f'Circle with radius {radius}')

            # Color gradient over time
            norm = mcolors.Normalize(vmin=0, vmax=len(allxpos))
            cmap = sns.color_palette("Spectral", as_cmap=True)
            colors = [cmap(norm(i)) for i in range(len(allxpos) - 1)]

            # Plot arrows between successive points
            for i in range(len(allxpos) - 1):
                ax.arrow(
                    allxpos[i], allypos[i],
                    allxpos[i + 1] - allxpos[i],
                    allypos[i + 1] - allypos[i],
                    head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i], alpha=1
                )

            # Colorbar legend for time (s)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Time (s)')

            plt.title("Physical Path")
            plt.grid(False)
            plt.xlim(-radius, radius)
            plt.ylim(-radius, radius)
            plt.savefig("/Users/ezhu/Downloads/cs-physical-path.png", dpi=SAVE_DPI, bbox_inches='tight')
            plt.show()

        processedvel.append(tempvel)

    # ----- Aggregate to arrays/dataframes -----
    phalf = np.concatenate(processedpos, axis=0)
    phalf = np.reshape(phalf, [phalf.shape[0] * phalf.shape[1], 2])

    vhalf = np.concatenate(processedvel, axis=0)
    vhalf = np.reshape(vhalf, [vhalf.shape[0] * vhalf.shape[1], 2])

    # ----- Quick position heatmaps (two variants) -----
    center = (0, 0)
    theta = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
    xc = radius * np.cos(theta)
    yc = radius * np.sin(theta)

    # Small square heatmap
    plt.figure(figsize=(3, 3))
    plt.rcParams['figure.dpi'] = FIG_DPI
    plt.plot(xc, yc, label=f'Circle with radius {radius}')
    plt.hist2d(
        phalf[:, 0], Y_FLIP * phalf[:, 1],
        bins=(BINS_2D, BINS_2D),
        range=[[-radius, radius], [-radius, radius]],
        cmap=sns.color_palette(COLORMAP_LIGHTB, as_cmap=True),
        density=True,
        vmin=0,
        vmax=0.05
    )
    plt.savefig("/Users/ezhu/Downloads/cs-histogram.png", dpi=SAVE_DPI, bbox_inches='tight')
    plt.show()

    # Square-proportion heatmap with colorbar
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams['figure.dpi'] = FIG_DPI
    ax.plot(xc, yc, label=f'Circle with radius {radius}')
    h = ax.hist2d(
        phalf[:, 0],
        Y_FLIP * phalf[:, 1],
        bins=(BINS_2D, BINS_2D),
        range=[[-radius, radius], [-radius, radius]],
        cmap=sns.color_palette(COLORMAP_LIGHTB, as_cmap=True),
        density=True,
        vmin=0,
        vmax=0.05,
    )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    fig.colorbar(h[-1], ax=ax, label='Frequency')
    plt.savefig("/Users/ezhu/Downloads/cs-histogram-square.png", dpi=SAVE_DPI, bbox_inches='tight')
    plt.show()

    # ----- Assemble DataFrames & derived quantities -----
    phalf = pd.DataFrame(phalf, columns=['x', 'y'])
    phalf['r'] = np.sqrt(phalf['x'] ** 2 + phalf['y'] ** 2)
    phalf['y'] = Y_FLIP * phalf['y']
    phalf['theta'] = np.arctan2(Y_FLIP * phalf['x'], phalf['y'])

    vhalf = pd.DataFrame(vhalf, columns=['vx', 'vy'])
    vhalf['spd'] = np.sqrt(vhalf['vx'] ** 2 + vhalf['vy'] ** 2)
    vhalf['vy'] = Y_FLIP * vhalf['vy']
    vhalf['vtheta'] = np.arctan2(Y_FLIP * vhalf['vx'], vhalf['vy'])

    print('avg spd: ' + str(np.mean(vhalf['spd'])))

    half_df = pd.concat([phalf, vhalf], axis=1)
    half_df['vrx'] = half_df['vx'] * (half_df['x'] * half_df['vx'] + half_df['y'] * half_df['vy']) / (half_df['r'] * half_df['spd'])
    half_df['vry'] = half_df['vy'] * (half_df['x'] * half_df['vx'] + half_df['y'] * half_df['vy']) / (half_df['r'] * half_df['spd'])
    half_df['vr'] = half_df['spd'] * (half_df['x'] * half_df['vx'] + half_df['y'] * half_df['vy']) / (half_df['r'] * half_df['spd'])
    half_df['spd_r'] = np.abs(half_df['vr'])
    half_df['vtx'] = half_df['vx'] - half_df['vrx']
    half_df['vty'] = half_df['vy'] - half_df['vry']
    half_df['spd_t'] = np.sqrt(half_df['vtx'] ** 2 + half_df['vty'] ** 2)

    # Fixed parenthesis bug here (sin term)
    phi_temp = np.arccos(
        (-np.cos(half_df['theta']) * half_df['vx'] - np.sin(half_df['theta']) * half_df['vy']) / half_df['spd']
    )
    half_df['phi'] = phi_temp

    # from earlier accumulation
    half_df['refl_prop'] = refl_prop

    # ----- Turning-window detection (near wall) -----
    turn_times = []
    turns = []
    temp_counter = 0
    min_phi = 0
    max_phi = np.pi
    for _, row in half_df.iterrows():
        if temp_counter == 0 and row['r'] > NEAR_WALL_FRACTION * radius and row['phi'] < TURN_PHI_WINDOW:
            max_phi = row['phi'] + np.pi / 2
            temp_counter += 1
        elif temp_counter == 0 and row['r'] > NEAR_WALL_FRACTION * radius and row['phi'] > np.pi - TURN_PHI_WINDOW:
            min_phi = row['phi'] - np.pi / 2
            temp_counter += 1
        elif temp_counter > 0 and (row['phi'] >= min_phi) and (row['phi'] <= max_phi):
            temp_counter += 1
        elif temp_counter > 0:
            turns.append([row['x'], row['y']])
            turn_times.append(temp_counter)
            temp_counter = 0
            min_phi = 0
            max_phi = np.pi

    tt_avg.append(np.mean(turn_times))
    tt_std.append(np.std(turn_times))

    # ----- Optional per-age individual plots -----
    color = 'red' if (x % 10 == 0) else 'blue'
    if indiv == 'Y':
        f, ax = plt.subplots(figsize=(10, 8))
        corr = half_df.corr()
        sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    vmin=-1.0, vmax=1.0, square=True, ax=ax)
        plt.show()

        nearwall_df = half_df[half_df['r'] > (NEAR_WALL_FRACTION * radius)]
        sns.histplot(data=nearwall_df, x='phi', stat='percent', bins=10,
                     binrange=[0, np.pi/2], color=color, alpha=0.5)
        plt.xlabel('Phi')
        plt.ylabel('Percent')
        plt.ylim(0, 50)
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Phi Histogram for 1 Fish Sanded Tank ' + str(n) + 'dpf')
        else:
            plt.title('Phi Histogram for 1 Fish Clear Tank ' + str(x) + 'dpf')
        plt.show()

        sns.histplot(data=half_df, x='theta', stat='percent', bins=20,
                     binrange=[-np.pi, np.pi], color=color, alpha=0.5)
        plt.xlabel('Theta')
        plt.ylabel('Percent')
        plt.ylim(0, 12.5)
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Theta Histogram for 1 Fish Sanded Tank ' + str(n) + 'dpf')
        else:
            plt.title('Theta Histogram for 1 Fish Clear Tank ' + str(x) + 'dpf')
        plt.show()

        sns.histplot(data=half_df, x='spd_r', stat='percent', bins=10,
                     binrange=[0, 2.5], color=color, alpha=0.5)
        plt.xlabel('Radial Speed')
        plt.ylabel('Percent')
        plt.ylim(0, 100)
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Radial Speed Histogram for 1 Fish Sanded Tank ' + str(n) + 'dpf')
        else:
            plt.title('Radial Speed Histogram for 1 Fish Clear Tank ' + str(x) + 'dpf')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='spd_r', s=5, color=color, alpha=0.5)
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Relationship between Radial Speed and Radial Position for Sanded ' + str(n) + 'dpf')
        else:
            plt.title('Relationship between Radial Speed and Radial Position for Clear ' + str(x) + 'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Radial Speed')
        plt.xlim(0, radius)
        plt.ylim(0, 3)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='spd', s=5, color=color, alpha=0.5)
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Relationship between Speed and Radial Position for Sanded ' + str(n) + 'dpf')
        else:
            plt.title('Relationship between Speed and Radial Position for Clear ' + str(x) + 'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Radial Speed')
        plt.xlim(0, radius)
        plt.ylim(0, 3)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='vr', s=5, color=color, alpha=0.5)
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Relationship between Radial Velocity and Radial Position for Sanded ' + str(n) + 'dpf')
        else:
            plt.title('Relationship between Radial Velocity and Radial Position for Clear ' + str(x) + 'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Radial Velocity')
        plt.xlim(0, radius)
        plt.ylim(-3, 3)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='phi', s=5, color=color, alpha=0.5)
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Relationship between Wall Angle and Radial Position for Sanded ' + str(n) + 'dpf')
        else:
            plt.title('Relationship between Wall Angle and Radial Position for Clear ' + str(x) + 'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Angle to Wall')
        plt.xlim(0, radius)
        plt.grid(True)
        plt.show()

    # ----- Turning-location heatmap -----
    if len(turns) > 0:
        turns = np.array(turns)
        plt.hist2d(
            turns[:, 0], turns[:, 1],
            bins=(BINS_2D, BINS_2D),
            range=[[-1 * radius, radius], [-1 * radius, radius]],
            cmap=sns.color_palette(COLORMAP_LIGHTB, as_cmap=True),
            density=True,
            vmin=0, vmax=0.04
        )
        plt.xlabel('X-bins')
        plt.ylabel('Y-bins')
        if x % 10 == 0:
            n = int(x / 10)
            plt.title('Heatmap for Turning Location 1 Fish Sanded Tank ' + str(n) + 'dpf')
        else:
            plt.title('Heatmap for Turning Location 1 Fish Clear Tank ' + str(x) + 'dpf')
        plt.colorbar(label='Frequency')
        plt.show()

    outputs.append(half_df.reset_index(drop=True))

# ----- Summary: mean turning time over ages -----
dpf = [7, 14, 21]
plt.errorbar(x=dpf, y=tt_avg, yerr=tt_std, fmt='o', color=('red' if arr[0] % 10 == 0 else 'blue'))
plt.xlabel('Days Post Fertilization')
plt.ylabel('Turning Time Along Wall')
if arr[0] % 10 == 0:
    plt.title('Mean Turning Time over dpf for Sanded')
else:
    plt.title('Mean Turning Time over dpf for Clear')
plt.show()

# Tag each per-age frame, then combine (avoid duplicate indices for seaborn)
for i in range(len(outputs)):
    if arr[i] % 10 == 0:
        outputs[i]['Age'] = f"{int(arr[i]/10)}dpf"
    else:
        outputs[i]['Age'] = f"{arr[i]}dpf"

combined_df = pd.concat(outputs, ignore_index=True)

# ----- Combined comparisons -----
sns.histplot(
    data=combined_df, x='r', stat='percent', hue='Age',
    bins=BINS_2D, binrange=[0, 10],
    palette=sns.color_palette(palette='YlGnBu_r'),
    alpha=0.75, multiple='dodge', common_norm=False
)
x0 = arr[0]
plt.title('Distance From Center for 1 Fish Sanded Tank Over Time' if x0 % 10 == 0
          else 'Distance From Center for 1 Fish Clear Tank Over Time')
plt.show()

sns.histplot(
    data=combined_df, x='spd', stat='percent', hue='Age',
    bins=10, binrange=[0, 5],
    palette=sns.color_palette(palette='YlGnBu_r'),
    alpha=0.75, multiple='dodge', common_norm=False
)
plt.title('Speed for 1 Fish Sanded Tank Over Time' if x0 % 10 == 0
          else 'Speed for 1 Fish Clear Tank Over Time')
plt.show()

sns.histplot(
    data=combined_df, x='spd_r', stat='percent', hue='Age',
    bins=10, binrange=[0, 2.5],
    palette=sns.color_palette(palette='YlGnBu_r'),
    alpha=0.75, multiple='dodge', common_norm=False
)
plt.title('Radial Speed for 1 Fish Sanded Tank Over Time' if x0 % 10 == 0
          else 'Radial Speed for 1 Fish Clear Tank Over Time')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=combined_df, x='r', y='spd_r', s=5,
    hue='Age', palette=sns.color_palette(palette='YlGnBu_r'),
    alpha=0.25
)
plt.title('Relationship between Radial Speed and Radial Position')
plt.xlabel('Radial Position')
plt.ylabel('Radial Speed')
plt.grid(True)
plt.show()

# ----- Outlined radial histogram (percent per bin) -----
plt.figure(figsize=(10, 6))
colors = ['blue', 'purple', 'red']
bin_edges = np.linspace(0, radius, 11)

for i, output in enumerate(outputs):
    data = output['r'].dropna().values
    if data.size == 0:
        continue
    weights = np.ones_like(data, dtype=float) * (100.0 / data.size)
    plt.hist(
        data,
        bins=bin_edges,
        weights=weights,
        histtype='bar',
        linewidth=2.5,
        edgecolor=colors[i % len(colors)],
        facecolor='none',
        fill=False,
        label=(f"{int(arr[i]/10)}dpf" if arr[i] % 10 == 0 else f"{arr[i]}dpf"),
        alpha=0.4
    )

plt.xlabel('Radial Position (cm)')
plt.ylabel('Percentage (%)')
plt.xlim([0, radius])
plt.ylim([0, 60])
plt.legend()
plt.savefig("/Users/ezhu/Downloads/radial_histogram.png", dpi=SAVE_DPI, bbox_inches='tight')
plt.show()

# ----- Overlayed Phi (near-wall) histogram (outlined) -----
plt.figure(figsize=(10, 6))
colors = ['blue', 'purple', 'red']
PHI_BIN_EDGES = np.linspace(0, np.nextafter(np.pi/2, 0), 9)

for i, output in enumerate(outputs):
    nearwall_df = output[output['r'] > NEAR_WALL_FRACTION * radius]
    data = nearwall_df['phi'].dropna().values
    if data.size == 0:
        continue
    mask = (data >= 0) & (data < np.pi/2)
    data = data[mask]

    sns.histplot(
        x=data,
        bins=PHI_BIN_EDGES,
        stat='percent',
        element='step',
        fill=False,
        linewidth=2.5,
        alpha=0.4,
        color=colors[i % len(colors)],
        label=(f"{int(arr[i]/10)}dpf" if arr[i] % 10 == 0 else f"{arr[i]}dpf"),
        common_norm=False,
    )

plt.xlabel('Angle From Wall (rad)')
plt.ylabel('Percentage (%)')
plt.xlim([0, np.pi/2])
plt.ylim([0, 60])
plt.legend()
plt.savefig('/Users/ezhu/Downloads/phi_histogram_overlay-sanded-blind.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.show()