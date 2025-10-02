"""
This script loads single-fish idtracker.ai trajectories for half-sanded arenas (left/right contrast),
optionally for blind (TYR) fish, converts them to physical units, and produces:
  • Position heatmaps (arena-centered)
  • Distributions of r, speed, radial speed, and wall angle φ
  • Side-split analyses (x<0 vs x>0) for near-wall φ
  • Turning-time estimates near the boundary, separated by side

It validates user choices, aggregates sessions for 7/14/21 dpf, computes radial/tangential components,
and summarizes mean turning time by age for clear vs sanded halves. All adjustable parameters live in
CONFIG below; plots keep stylistic numbers inline.
"""

from pprint import pprint
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import trajectorytools as tt
import matplotlib.colors as mcolors
import sys

# ==========================
# CONFIGURATION (adjustable)
# ==========================
RADIUS = 5.0                 # cm, arena radius used for physical scaling and bounds
NEAR_WALL_FRACTION = 0.8     # r/RADIUS threshold considered "near wall"
TURN_PHI_WINDOW = 0.4        # radians, φ window when entering near-wall band
BINS_2D = 10                 # 2D heatmap bins per axis
CIRCLE_POINTS = 300          # resolution for boundary circle
FIG_DPI = 100                # on-screen figure DPI
SAVE_DPI = 3000              # exported PNG DPI
ANGLE_STEP = 0.01            # radians, sampling step for boundary angles
COLORMAP_LIGHTB = "light:b"  # seaborn palette name for 2D histograms

# idtracker.ai pixel geometry (px)
PX_CENTER = 1048
PX_DIAMETER = 2048
PX_HALF = 1024

# Coordinate convention
Y_FLIP = -1                  # multiply y by this to flip image to math coordinates

# Phi histogram bin edges (0 to pi/2, open upper edge)
PHI_BIN_EDGES = np.linspace(0, np.nextafter(np.pi/2, 0), 9)

# ----- User inputs -----
arr = [7, 14, 21]
indiv = input('Individual plots? (Y/N) : ').strip().upper()
blind = input('Blind fish? (Y/N) : ').strip().upper()

# Validate user inputs (exit on bad values)
if indiv not in ('Y', 'N'):
    print("Invalid input for 'Individual plots?'. Please enter 'Y' or 'N'. Exiting.")
    sys.exit(1)
if blind not in ('Y', 'N'):
    print("Invalid input for 'Blind fish?'. Please enter 'Y' or 'N'. Exiting.")
    sys.exit(1)

outputs = []

tt_avg_s, tt_std_s = [], []
tt_avg_c, tt_std_c = [], []
turns_sep = []

for x in arr:
    path = input('Path plots? (Y/N) : ').strip().upper()
    if path not in ('Y', 'N'):
        print("Invalid input for 'Path plots?'. Please enter 'Y' or 'N'. Exiting.")
        sys.exit(1)
    radius = RADIUS

    # ----- File lists per age -----
    if x == 7:
        file1 = "data/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
        file2 = "data/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
        file3 = "data/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
        file4 = "data/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
        file5 = "data/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"
        file6 = "data/07.02.24/session_1fish-1fps-15min-7dpf-half1/trajectories/validated.npy"
        file7 = "data/07.02.24/session_1fish-1fps-15min-7dpf-half2/trajectories/validated.npy"
        file8 = "data/07.02.24/session_1fish-1fps-15min-7dpf-half3/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5, file6, file7, file8]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-half1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-half2-crispr/trajectories/validated.npy"
            file8 = "data/10.17.24/session_1fish-1fps-15min-7dpf-half3-crispr/trajectories/validated.npy"
            files = [file1, file2, file3, file4, file5, file6, file7, file8]
    elif x == 14:
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half5/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-half1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-half2-crispr/trajectories/validated.npy"
            files = [file1, file2, file3, file4, file5, file6, file7]
    elif x == 21:
        file1 = "data/3.13.24/session_1fish15min1fps-half-1-21dpf/trajectories/validated.npy"
        file2 = "data/3.13.24/session_1fish15min1fps-half-2-21dpf/trajectories/validated.npy"
        file3 = "data/3.13.24/session_1fish15min1fps-half-3-21dpf/trajectories/validated.npy"
        file4 = "data/5.21.24/session_1fish-1fps-15min-21dpf-half-4/trajectories/validated.npy"
        file5 = "data/07.16.24/session_1fish-1fps-15min-21dpf-half1/trajectories/validated.npy"
        file6 = "data/07.16.24/session_1fish-1fps-15min-21dpf-half2/trajectories/validated.npy"
        file7 = "data/07.16.24/session_1fish-1fps-15min-21dpf-half3/trajectories/validated.npy"
        file8 = "data/07.16.24/session_1fish-1fps-15min-21dpf-half4/trajectories/validated.npy"
        file9 = "data/07.16.24/session_1fish-1fps-15min-21dpf-half5/trajectories/validated.npy"
        files = [file1, file2, file3, file4, file5, file6, file7, file8, file9]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-half1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-half2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-half3-crispr/trajectories/validated.npy"
            files = [file1, file2, file3]

    # ----- Helpers: counting and file I/O -----
    def count_ones(array):
        count = 0
        for num in array:
            if num == 1:
                count += 1
        return count

    def openfile(file, sigma=1):
        """Open a trajectory file with optional Gaussian smoothing (sigma in frames)."""
        tr = tt.Trajectories.from_idtrackerai(
            file,
            interpolate_nans=True,
            smooth_params={'sigma': sigma}
        )
        return tr

    trs = [openfile(file) for file in files]

    # ----- Preprocess trajectories: center, units, diagnostics -----
    def processtr(tr):
        center, radiustr = tr.estimate_center_and_radius_from_locations(in_px=True)
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
        """Estimate fraction of boundary points visible as reflection, given position and heading."""
        mag = np.sqrt(xposition**2 + yposition**2)
        magv = np.sqrt(xvelocity**2 + yvelocity**2)
        _ = 2 * (radius - mag)  # geometric context (not used numerically below)

        angles = np.arange(0, 2*np.pi, ANGLE_STEP)
        xbound = radius * np.cos(angles)
        ybound = radius * np.sin(angles)
        labels = np.zeros(len(angles))

        ANGLE_MIN = np.pi/2
        ANGLE_MAX = 3*np.pi/2
        THETA_MIN = 0.85
        THETA_MAX = 2.29
        PHI_MAX   = 2.958

        for i in range(len(angles)):
            magd = np.sqrt((xbound[i]-xposition)**2 + (ybound[i]-yposition)**2)
            theta = np.arccos((xbound[i]*(xbound[i]-xposition) + ybound[i]*(ybound[i]-yposition)) / (radius*magd))
            phi   = np.arccos((xvelocity*(xbound[i]-xposition) + yvelocity*(ybound[i]-yposition)) / (magv*magd))
            if (angles[i] > ANGLE_MIN) and (angles[i] < ANGLE_MAX) and (theta > THETA_MIN) and (theta < THETA_MAX) and (phi < PHI_MAX):
                labels[i] = 1
        return count_ones(labels) / len(labels)

    refl_prop = []

    # ----- Process and (optionally) plot per-file paths -----
    def border_turning(tr):
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

    processedpos, processedvel = [], []
    for temp in trs:
        processed_temp = processtr(temp)
        border_turning(processed_temp)

        offset = processed_temp.params['_center'] - np.array([PX_CENTER, PX_CENTER])
        temppos = processed_temp.s * processed_temp.params['length_unit'] * (2 * radius / PX_DIAMETER) + offset * radius / PX_HALF
        tempvel = processed_temp.v * (processed_temp.params['length_unit'] / processed_temp.params['time_unit']) * (2 * radius / PX_DIAMETER)
        processedpos.append(temppos)
        processedvel.append(tempvel)

        if path == 'Y':
            all_positions = np.reshape(np.array(temppos), (-1, 2))
            allxpos = all_positions[:, 0]
            allypos = Y_FLIP * all_positions[:, 1]

            plt.figure(figsize=(6, 6))
            plt.rcParams['figure.dpi'] = FIG_DPI
            center = (0, 0)
            theta = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
            xc = center[0] + radius * np.cos(theta)
            yc = center[1] + radius * np.sin(theta)
            plt.plot(xc, yc, label=f'Circle with radius {radius}')

            norm = mcolors.Normalize(vmin=0, vmax=len(allxpos))
            cmap = sns.color_palette("Spectral", as_cmap=True)
            colors = [cmap(norm(i)) for i in range(len(allxpos) - 1)]

            for i in range(len(allxpos) - 1):
                plt.arrow(
                    allxpos[i], allypos[i],
                    allxpos[i+1] - allxpos[i], allypos[i+1] - allypos[i],
                    head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i], alpha=1
                )

            plt.title("Physical Path")
            plt.grid(False)
            plt.xlim(-radius, radius)
            plt.ylim(-radius, radius)
            plt.savefig("/Users/ezhu/Downloads/cs-physical-path.png", dpi=SAVE_DPI, bbox_inches='tight')
            plt.show()

    # ----- Aggregate to arrays -----
    phalf = np.concatenate(processedpos, axis=0)
    phalf = np.reshape(phalf, [phalf.shape[0] * phalf.shape[1], 2])

    center = (0, 0)
    theta = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
    xc = center[0] + radius * np.cos(theta)
    yc = center[1] + radius * np.sin(theta)

    # ----- Position heatmap (small square) -----
    plt.figure(figsize=(3, 3))
    plt.rcParams['figure.dpi'] = FIG_DPI
    plt.plot(xc, yc, label=f'Circle with radius {radius}')
    plt.hist2d(
        phalf[:, 0], Y_FLIP * phalf[:, 1],
        bins=(BINS_2D, BINS_2D),
        range=[[-radius, radius], [-radius, radius]],
        cmap=sns.color_palette(COLORMAP_LIGHTB, as_cmap=True),
        density=True,
        vmin=0, vmax=0.05
    )
    plt.savefig("/Users/ezhu/Downloads/h-histogram.png", dpi=SAVE_DPI, bbox_inches='tight')
    plt.show()

    vhalf = np.concatenate(processedvel, axis=0)
    vhalf = np.reshape(vhalf, [vhalf.shape[0] * vhalf.shape[1], 2])

    # ----- DataFrames & derived quantities -----
    phalf = pd.DataFrame(phalf, columns=['x', 'y'])
    phalf['r'] = np.sqrt(phalf['x']**2 + phalf['y']**2)
    phalf['y'] = Y_FLIP * phalf['y']
    phalf['theta'] = np.arctan2(Y_FLIP * phalf['x'], phalf['y'])

    vhalf = pd.DataFrame(vhalf, columns=['vx', 'vy'])
    vhalf['spd'] = np.sqrt(vhalf['vx']**2 + vhalf['vy']**2)
    vhalf['vy'] = Y_FLIP * vhalf['vy']
    vhalf['vtheta'] = np.arctan2(Y_FLIP * vhalf['vx'], vhalf['vy'])

    plt.figure()
    plt.title('avg spd: ' + str(np.mean(vhalf['spd'])))
    plt.hist(vhalf['spd'], bins=20, range=[0, 2], density=True)
    plt.show()

    half_df = pd.concat([phalf, vhalf], axis=1)
    half_df['vrx']   = half_df['vx']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    half_df['vry']   = half_df['vy']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    half_df['vr']    = half_df['spd']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    half_df['spd_r'] = np.abs(half_df['vr'])
    half_df['vtx']   = half_df['vx']-half_df['vrx']
    half_df['vty']   = half_df['vy']-half_df['vry']
    half_df['spd_t'] = np.sqrt(half_df['vtx']**2+half_df['vty']**2)

    # Fixed parenthesis bug in φ
    phi_temp = np.arccos(( -np.cos(half_df['theta'])*half_df['vx'] - np.sin(half_df['theta'])*half_df['vy'] )/half_df['spd'])
    phi_temp = np.where(phi_temp > np.pi/2, np.pi - phi_temp, phi_temp)
    half_df['phi'] = phi_temp

    half_df['refl_prop'] = refl_prop
    half_df['side'] = half_df['theta'].apply(lambda t: 'clear' if t > 0 else 'sanded')

    # ----- Turning-window detection (near wall) -----
    turn_times_s, turn_times_c = [], []
    turns = []
    temp_counter = 0
    min_phi = 0
    max_phi = np.pi
    for _, row in half_df.iterrows():
        if temp_counter == 0 and row['r'] > NEAR_WALL_FRACTION*radius and row['phi'] < TURN_PHI_WINDOW:
            current = row['side']
            max_phi = row['phi'] + np.pi/2
            temp_counter += 1
        elif temp_counter == 0 and row['r'] > NEAR_WALL_FRACTION*radius and row['phi'] > np.pi - TURN_PHI_WINDOW:
            current = row['side']
            min_phi = row['phi'] - np.pi/2
            temp_counter += 1
        elif temp_counter > 0 and (row['phi'] >= min_phi) and (row['phi'] <= max_phi):
            temp_counter += 1
        elif temp_counter > 0:
            turns.append([row['x'], row['y']])
            if current == 'clear':
                turn_times_c.append(temp_counter)
            else:
                turn_times_s.append(temp_counter)
            temp_counter = 0
            min_phi = 0
            max_phi = np.pi

    turns_sep.append(turns)
    tt_avg_s.append(np.mean(turn_times_s))
    tt_avg_c.append(np.mean(turn_times_c))
    tt_std_s.append(np.std(turn_times_s))
    tt_std_c.append(np.std(turn_times_c))

    # ----- Individual plots (optional) -----
    if indiv == 'Y':
        sns.histplot(
            data=half_df, x='theta', stat='percent', bins=20, binrange=[-np.pi, np.pi],
            hue='side', palette={'clear':'blue','sanded':'red'}, alpha=0.5, multiple='dodge', common_norm=True
        )
        plt.xlabel('Theta'); plt.ylabel('Percent'); plt.ylim(0, 12.5)
        plt.title(f'Theta Histogram for 1 Fish HalfSanded Tank {x}dpf')
        plt.show()

        sns.histplot(
            data=half_df, x='spd_r', stat='percent', bins=10, binrange=[0, 2.5],
            hue='side', palette={'clear':'blue','sanded':'red'}, alpha=0.5, multiple='dodge', common_norm=False
        )
        plt.xlabel('Radial Speed'); plt.ylabel('Percent'); plt.ylim(0, 100)
        plt.title(f'Radial Speed Histogram for 1 Fish Half Sanded Tank {x}dpf')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='phi', hue='side', palette={'clear':'blue','sanded':'red'}, s=5, alpha=0.5)
        plt.title(f'Relationship between Wall Angle and Radial Position Half Sanded {x}dpf')
        plt.xlabel('Radial Position'); plt.ylabel('Angle to Wall')
        plt.xlim(0, radius); plt.grid(True); plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='spd_r', hue='side', palette={'clear':'blue','sanded':'red'}, s=5, alpha=0.5)
        plt.title(f'Relationship between Radial Speed and Radial Position Half Sanded {x}dpf')
        plt.xlabel('Radial Position'); plt.ylabel('Radial Speed')
        plt.xlim(0, radius); plt.ylim(0, 3); plt.grid(True); plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='spd', hue='side', palette={'clear':'blue','sanded':'red'}, s=5, alpha=0.5)
        plt.title(f'Relationship between Radial Speed and Radial Position Half Sanded {x}dpf')
        plt.xlabel('Radial Position'); plt.ylabel('Speed')
        plt.xlim(0, radius); plt.ylim(0, 3); plt.grid(True); plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='vr', hue='side', palette={'clear':'blue','sanded':'red'}, s=5, alpha=0.5)
        plt.title(f'Relationship between Radial Velocity and Radial Position Half Sanded {x}dpf')
        plt.xlabel('Radial Position'); plt.ylabel('Radial Velocity')
        plt.xlim(0, radius); plt.ylim(-3, 3); plt.grid(True); plt.show()

        turns_arr = np.array(turns)
        if turns_arr.size:
            plt.hist2d(
                turns_arr[:, 0], turns_arr[:, 1],
                bins=(BINS_2D, BINS_2D),
                range=[[-radius, radius], [-radius, radius]],
                cmap=sns.color_palette(COLORMAP_LIGHTB, as_cmap=True),
                density=True, vmin=0, vmax=0.04
            )
            plt.xlabel('X-bins'); plt.ylabel('Y-bins')
            plt.title(f'Heatmap for Turning Location 1 Fish Half Sanded Tank {x}dpf')
            plt.colorbar(label='Frequency'); plt.show()

    outputs.append(half_df.reset_index(drop=True))

# ----- Summary turning time over ages -----
plt.errorbar(x=arr, y=tt_avg_s, yerr=tt_std_s, fmt='o', color='red',  alpha=0.5, label='sanded')
plt.errorbar(x=arr, y=tt_avg_c, yerr=tt_std_c, fmt='o', color='blue', alpha=0.5, label='clear')
plt.legend()
plt.xlabel('Days Post Fertilization')
plt.ylabel('Turning Time Along Wall')
plt.title('Mean Turning Time over dpf (Half-Sanded)')
plt.show()

# ----- Combine & compare across ages -----
for i in range(len(outputs)):
    outputs[i]['Age'] = f"{arr[i]}dpf"
combined_df = pd.concat(outputs, ignore_index=True)

sns.histplot(
    data=combined_df, x='r', stat='percent', hue='Age',
    bins=BINS_2D, binrange=[0, 2*RADIUS],
    palette=sns.color_palette(palette='YlGnBu_r'),
    alpha=0.75, multiple='dodge', common_norm=False
)
plt.title('Distance From Center for 1 Fish Half Sanded Tank Over Time')
plt.show()

# ----- Phi overlays (near wall) -----
# (1) x < 0
plt.figure(figsize=(10, 6))
colors = ['blue', 'purple', 'red']
for i, output in enumerate(outputs):
    nearwall_df = output[(output['r'] > NEAR_WALL_FRACTION*RADIUS) & (output['x'] < 0)]
    data = nearwall_df['phi'].dropna().values
    if data.size == 0: 
        continue
    data = data[(data >= 0) & (data < np.pi/2)]
    sns.histplot(
        x=data, bins=PHI_BIN_EDGES, stat='percent',
        element='step', fill=False, linewidth=2.5, alpha=0.4,
        color=colors[i % len(colors)],
        label=(f\"{int(arr[i]/10)}dpf (x<0)\" if arr[i] % 10 == 0 else f\"{arr[i]}dpf (x<0)\"),
        common_norm=False,
    )
plt.xlabel('Angle From Wall (rad)'); plt.ylabel('Percentage (%)')
plt.xlim([0, np.pi/2]); plt.ylim([0, 60]); plt.legend()
plt.savefig('/Users/ezhu/Downloads/phi_histogram_overlay-half_xneg.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.show()

# (2) x > 0
plt.figure(figsize=(10, 6))
for i, output in enumerate(outputs):
    nearwall_df = output[(output['r'] > NEAR_WALL_FRACTION*RADIUS) & (output['x'] > 0)]
    data = nearwall_df['phi'].dropna().values
    if data.size == 0:
        continue
    data = data[(data >= 0) & (data < np.pi/2)]
    sns.histplot(
        x=data, bins=PHI_BIN_EDGES, stat='percent',
        element='step', fill=False, linewidth=2.5, alpha=0.4,
        color=colors[i % len(colors)],
        label=(f\"{int(arr[i]/10)}dpf (x>0)\" if arr[i] % 10 == 0 else f\"{arr[i]}dpf (x>0)\"),
        common_norm=False,
    )
plt.xlabel('Angle From Wall (rad)'); plt.ylabel('Percentage (%)')
plt.xlim([0, np.pi/2]); plt.ylim([0, 60]); plt.legend()
plt.savefig('/Users/ezhu/Downloads/phi_histogram_overlay-half_xpos.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.show()

# (3) combined x<0 dashed vs x>0 solid
plt.figure(figsize=(10, 6))
for i, output in enumerate(outputs):
    neg = output[(output['r'] > NEAR_WALL_FRACTION*RADIUS) & (output['x'] < 0)]['phi'].dropna().values
    pos = output[(output['r'] > NEAR_WALL_FRACTION*RADIUS) & (output['x'] > 0)]['phi'].dropna().values
    neg = neg[(neg >= 0) & (neg < np.pi/2)]
    pos = pos[(pos >= 0) & (pos < np.pi/2)]
    counts_neg, _ = np.histogram(neg, bins=PHI_BIN_EDGES)
    counts_pos, _ = np.histogram(pos, bins=PHI_BIN_EDGES)
    total_neg = neg.size if neg.size else 1
    total_pos = pos.size if pos.size else 1
    percent_neg = 100.0 * counts_neg / total_neg
    percent_pos = 100.0 * counts_pos / total_pos
    bin_centers = 0.5*(PHI_BIN_EDGES[:-1] + PHI_BIN_EDGES[1:])
    label_base = (f\"{int(arr[i]/10)}dpf\" if arr[i] % 10 == 0 else f\"{arr[i]}dpf\")
    plt.plot(bin_centers, percent_pos, drawstyle='steps-mid', linewidth=2.5,
             color=colors[i % len(colors)], alpha=0.4, label=f\"{label_base} (x>0)\")
    plt.plot(bin_centers, percent_neg, drawstyle='steps-mid', linewidth=2.5,
             color=colors[i % len(colors)], alpha=0.4, linestyle='--', label=f\"{label_base} (x<0)\")
plt.xlabel('Angle From Wall (rad)'); plt.ylabel('Percentage (%)')
plt.xlim([0, np.pi/2]); plt.ylim([0, 60]); plt.legend(ncol=2)
plt.savefig('/Users/ezhu/Downloads/phi_histogram_overlay-half_combined.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.show()

# (4) all data (no x split)
plt.figure(figsize=(10, 6))
for i, output in enumerate(outputs):
    data = output[output['r'] > NEAR_WALL_FRACTION*RADIUS]['phi'].dropna().values
    if data.size == 0:
        continue
    data = data[(data >= 0) & (data < np.pi/2)]
    sns.histplot(
        x=data, bins=PHI_BIN_EDGES, stat='percent',
        element='step', fill=False, linewidth=2.5, alpha=0.4,
        color=colors[i % len(colors)],
        label=(f\"{int(arr[i]/10)}dpf\" if arr[i] % 10 == 0 else f\"{arr[i]}dpf\"),
        common_norm=False,
    )
plt.xlabel('Angle From Wall (rad)'); plt.ylabel('Percentage (%)')
plt.xlim([0, np.pi/2]); plt.ylim([0, 60]); plt.legend()
plt.savefig('/Users/ezhu/Downloads/phi_histogram_overlay-half.png', dpi=SAVE_DPI, bbox_inches='tight')
plt.show()