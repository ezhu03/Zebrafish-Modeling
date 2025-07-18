import numpy as np
import matplotlib.pyplot as plt

data7 = np.load("speeddistribution7dpf.npy")
data7b = np.load("speeddistribution7dpf_blind.npy")
data14 = np.load("speeddistribution14dpf.npy")
data14b = np.load("speeddistribution14dpf_blind.npy")
data21 = np.load("speeddistribution21dpf.npy")
data21b = np.load("speeddistribution21dpf_blind.npy")

# NEW PLOT: Split p-values to one axis and D-statistics to another
plt.rcParams['figure.dpi'] = 100
# Create a new figure and a primary axis for p-values
fig, ax1 = plt.subplots(figsize=(4, 6))
# Create a secondary y-axis for D-statistics that shares the same x-axis

# --- Plot for p-values on ax1 ---
pos = [0.7, 2, 3.3]  # Positions for the violin plots
posb = [1.3, 2.6, 3.9]  # Positions for the blind data violin plots
ticks = [1,2.3,3.6]
parts = ax1.violinplot([data7, data14, data21],
                           vert=True, showextrema=False,
                           positions=pos)
# Set the violin colors for p-values (blue)
for pc in parts['bodies']:
    pc.set_facecolor('cornflowerblue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)

partsb = ax1.violinplot([data7b, data14b, data21b],
                           vert=True, showextrema=False,
                           positions=posb)
# Set the violin colors for blind data (orange)
for pc in partsb['bodies']:
    pc.set_facecolor('orange')
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)

# Overlay the box plots for p-values
boxplot = ax1.boxplot([data7, data14, data21],
                        positions=pos,
                        widths=0.1,
                        patch_artist=True,
                        showfliers=False,
                        vert=True)
for patch in boxplot['boxes']:
    patch.set_facecolor('white')
    patch.set_edgecolor('black')
for median in boxplot['medians']:
    median.set_color('black')
for whisker in boxplot['whiskers']:
    whisker.set_color('black')
for cap in boxplot['caps']:
    cap.set_color('black')

boxplotb = ax1.boxplot([data7b, data14b, data21b],
                        positions=posb,
                        widths=0.1,
                        patch_artist=True,
                        showfliers=False,
                        vert=True)
for patch in boxplotb['boxes']:
    patch.set_facecolor('white')
    patch.set_edgecolor('black')
for median in boxplotb['medians']:
    median.set_color('black')
for whisker in boxplotb['whiskers']:
    whisker.set_color('black')
for cap in boxplotb['caps']:
    cap.set_color('black')

ax1.set_xticks(ticks)
ax1.set_xticklabels(["7dpf", "14dpf", "21dpf"])
ax1.set_ylim(0, 2)
#ax1.set_ylabel("p-values", color="cornflowerblue")
ax1.tick_params(axis='y', labelcolor="black")
#ax1.set_title("Violin Plots with Different y-axis Limits for p-values and D-statistic")


from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor='cornflowerblue', edgecolor='black', alpha=0.5, label='Control'),
    Patch(facecolor='orange', edgecolor='black', alpha=0.5, label='Blind')
]
ax1.legend(handles=legend_patches, loc='upper right')
plt.tight_layout()
#plt.savefig("/Users/ezhu/Downloads/KSplot.png", dpi=3000, bbox_inches='tight')
plt.show()