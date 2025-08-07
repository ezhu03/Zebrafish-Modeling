import numpy as np
import matplotlib.pyplot as plt

data7 = np.load("modeling/data/speeddistribution/speeddistribution7dpf.npy")
data7b = np.load("modeling/data/speeddistribution/speeddistribution7dpf_blind.npy")
data14 = np.load("modeling/data/speeddistribution/speeddistribution14dpf.npy")
data14b = np.load("modeling/data/speeddistribution/speeddistribution14dpf_blind.npy")
data21 = np.load("modeling/data/speeddistribution/speeddistribution21dpf.npy")
data21b = np.load("modeling/data/speeddistribution/speeddistribution21dpf_blind.npy")

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
plt.savefig("/Users/ezhu/Downloads/speed_distribution_all.png", dpi=3000, bbox_inches='tight')
plt.show()

# KDE plot for control data
import seaborn as sns
plt.figure(figsize=(5, 4))
sns.kdeplot(data7, color='blue', label='7 dpf', linewidth=5, alpha=0.4)
sns.kdeplot(data14, color='purple', label='14 dpf', linewidth=5, alpha=0.4)
sns.kdeplot(data21, color='red', label='21 dpf', linewidth=5, alpha=0.4)
plt.xlabel('Speed')
plt.ylabel('Density')
plt.legend(title='Control')
plt.tight_layout()
plt.savefig("/Users/ezhu/Downloads/speed_distribution_control.png", dpi=3000, bbox_inches='tight')
plt.show()

# KDE plot for blind data
plt.figure(figsize=(5, 4))
sns.kdeplot(data7b, color='blue', label='7 dpf', linewidth=5, alpha=0.4)
sns.kdeplot(data14b, color='purple', label='14 dpf', linewidth=5, alpha=0.4)
sns.kdeplot(data21b, color='red', label='21 dpf', linewidth=5, alpha=0.4)
plt.xlabel('Speed')
plt.ylabel('Density')
plt.legend(title='Blind')
plt.tight_layout()
plt.savefig("/Users/ezhu/Downloads/speed_distribution_blind.png", dpi=3000, bbox_inches='tight')
plt.show()