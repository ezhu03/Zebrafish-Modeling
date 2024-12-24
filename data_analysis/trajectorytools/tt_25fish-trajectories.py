from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# trajectorytools needs to be installed. To install,
# pip install trajectorytools or follow the instructions at
# https://gitlab.com/polavieja_lab/trajectorytools
import trajectorytools as tt

trajectories_file_path = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_25fish-2fps-5min-21dpf-sanded/trajectories/validated.npy"

trajectories_dict: dict = np.load(trajectories_file_path, allow_pickle=True).item()

print(f"Content of the dictionary: {list(trajectories_dict.keys())}")
trajectories: np.ndarray = trajectories_dict["trajectories"]
print(
    "Trajectories from",
    trajectories_dict.get("video_paths") or trajectories_dict.get("video_path"),
)
print("Number of frames", trajectories.shape[0])
print("Number of individuals", trajectories.shape[1])
print(
    "X range:",
    np.nanmin(trajectories[..., 0]),
    np.nanmax(trajectories[..., 0]),
    "pixels",
)
print(
    "Y range:",
    np.nanmin(trajectories[..., 1]),
    np.nanmax(trajectories[..., 1]),
    "pixels",
)
print(trajectories_dict["frames_per_second"], "frames per second")
print(trajectories_dict["body_length"], "pixels per animal")

print("Loading trajectories from: ", trajectories_file_path)
tr = tt.Trajectories.from_idtrackerai(trajectories_file_path)
print("Positions array shape ", tr.s.shape)
print("Velocities array shape ", tr.v.shape)
print("Accelerations array shape ", tr.a.shape)
print("Positions:")
print("X range:", np.nanmin(tr.s[..., 0]), np.nanmax(tr.s[..., 0]), "pixels")
print("Y range:", np.nanmin(tr.s[..., 1]), np.nanmax(tr.s[..., 1]), "pixels")
print("Velcities:")
print("X range:", np.nanmin(tr.v[..., 0]), np.nanmax(tr.v[..., 0]), "pixels/frame")
print("Y range:", np.nanmin(tr.v[..., 1]), np.nanmax(tr.v[..., 1]), "pixels/frame")
print("Accelerations:")
print("X range:", np.nanmin(tr.a[..., 0]), np.nanmax(tr.a[..., 0]), "pixels/frame^2")
print("Y range:", np.nanmin(tr.a[..., 1]), np.nanmax(tr.a[..., 1]), "pixels/frame^2")
pprint(tr.params)

key_point_in_frame = np.asarray([1048, 1048])  # point in pixels
tr.origin_to(key_point_in_frame)

# Let assume that 50 pixels in the video frame are 1 cm.
tr.new_length_unit(200, "cm")

# Since we have the frames per second stored int the tr.params dictionary we will use them to
tr.new_time_unit(tr.params["frame_rate"], "s")



fig, ax_trajectories = plt.subplots(figsize=(5, 5))
time_range = (0, 59)
# SET HERE THE RANGE IN SECONDS FOR WHICH YOU WANT TO PLOT THE POSITIONS
frame_range = range(
    time_range[0] * tr.params["frame_rate"], time_range[1] * tr.params["frame_rate"], 1
)

ax_trajectories.plot(tr.s[frame_range, :, 0], tr.s[frame_range, :, 1])
ax_trajectories.set(
    aspect="equal", title="Trajectories", xlabel="X (BL)", ylabel="Y (BL)"
)
fig.tight_layout()
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

'''for i in range(25):
    fig, ax_trajectories = plt.subplots(figsize=(5, 5))
    time_range = (0, 59)
    # SET HERE THE RANGE IN SECONDS FOR WHICH YOU WANT TO PLOT THE POSITIONS
    frame_range = range(
        time_range[0] * tr.params["frame_rate"], time_range[1] * tr.params["frame_rate"], 1
    )

    ax_trajectories.plot(tr.s[frame_range, i, 0], tr.s[frame_range, i, 1])
    ax_trajectories.set(
        aspect="equal", title="Trajectories", xlabel="X (BL)", ylabel="Y (BL)"
    )
    fig.tight_layout()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
    plt.show()'''

fig, ax = plt.subplots(4, figsize=(10, 10), sharex=True)
# SET HERE THE RANGE IN SECONDS FOR WHICH YOU WANT TO PLOT THE VARIABLES
time_range = (0, 59)
frame_range = np.arange(
    time_range[0] * tr.params["frame_rate"], time_range[1] * tr.params["frame_rate"], 1
)

ax[0].plot(frame_range / tr.params["frame_rate"], tr.distance_to_origin[frame_range])
ax[1].plot(frame_range / tr.params["frame_rate"], tr.speed[frame_range])
ax[2].plot(frame_range / tr.params["frame_rate"], tr.acceleration[frame_range])
ax[3].plot(frame_range / tr.params["frame_rate"], tr.curvature[frame_range])

ax[0].set(ylabel="Distance to origin (BL)", ylim=(0, None))
ax[1].set(ylabel="Speed (BL/s)")
ax[2].set(ylabel="Acceleration ($BL/s^2$)")
ax[3].set(xlabel="t (s)", ylabel="Curvature ($1/BL$)", ylim=(-30, 30))

fig.tight_layout()
plt.show()

'''v_max = 0
min_x, max_x = np.nanmin(tr.s[..., 0]), np.nanmax(tr.s[..., 0])
min_y, max_y = np.nanmin(tr.s[..., 1]), np.nanmax(tr.s[..., 1])
X, Y = np.mgrid[min_x:max_x:50j, min_y:max_y:50j]
positions = np.vstack([X.ravel(), Y.ravel()])
G_kernels = []
for i in range(tr.number_of_individuals):
    print("computing Gaussian kernel for fish ", i)
    values = tr.s[~np.isnan(tr.s[:, i, 0]), i, :].T
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    G_kernels.append(Z)
    v_max = max(v_max, G_kernels[i].max())

fig, ax_arr = plt.subplots(
    1, tr.number_of_individuals, sharex=True, sharey=True, figsize=(90, 8)
)
fig.suptitle("Positions")

for i, ax in enumerate(ax_arr):
    ax.imshow(
        np.rot90(G_kernels[i]),
        cmap="jet",
        extent=[min_x, max_x, min_y, max_y],
        vmin=0,
        vmax=v_max,
    )
    ax.set(
        title=f"Fish {i}",
        xlabel="X position (BL)",
        aspect="equal",
    )
ax_arr[0].set(ylabel="Y position (BL)")
fig.tight_layout()
plt.show()'''

def histogram(data: np.ndarray, nbins: int, label: str):
    figv, ax_hist = plt.subplots(
        5, int(tr.number_of_individuals/5), figsize=(15, 3), sharey=True
    )
    for focal, ax in enumerate(ax_hist):
        ax.hist(data[:, focal][~np.isnan(data[:, focal])], nbins, density=True)
        ax.set(title=f"Fish {focal + 1}", xlabel=label)
    ax_hist[0].set(ylabel="PDF")
    figv.tight_layout()
    return figv, ax_hist


hist2d: list[np.ndarray] = []

for focal in range(tr.number_of_individuals):
    hist2d.append(
        np.histogram2d(
            tr.s[:, focal, 0][~np.isnan(tr.s[:, focal, 0])],
            tr.s[:, focal, 1][~np.isnan(tr.s[:, focal, 1])],
            25,
        )[0]
    )
vmax = np.asarray(hist2d).max()

# Plot distributions of positions in the arena
figv, ax_hist = plt.subplots(
    1, tr.number_of_individuals, figsize=(10, 3), sharey=True, sharex=True
)
min_x, max_x = np.nanmin(tr.s[..., 0]), np.nanmax(tr.s[..., 0])
min_y, max_y = np.nanmin(tr.s[..., 1]), np.nanmax(tr.s[..., 1])
for focal, ax in enumerate(ax_hist):
    ax.imshow(
        np.rot90(hist2d[focal]),
        interpolation="none",
        extent=[min_x, max_x, min_y, max_y],
        vmin=0,
        vmax=vmax,
    )
    ax.set(
        xlabel="X position (BL)",
        ylabel="Y position (BL)",
        title=f"Fish {focal + 1}",
        aspect="equal",
    )
figv.tight_layout()

# Plot distance to center of the arena histograms for each fish
nbins = np.linspace(0, np.nanmax(tr.distance_to_origin), 100)
figv, ax_flat = histogram(tr.distance_to_origin, nbins, "Distance to center (BL)")


# Plot speed histograms for each fish
nbins = np.linspace(0, 15, 100)  # ADJUST THE RANGE BINS TO YOUR DATA
figv, ax_flat = histogram(tr.speed, nbins, "Speed (BL/s)")


# Plot acceleration histograms for each fish
nbins = np.linspace(0, 150, 100)  # ADJUST THE RANGE BINS TO YOUR DATA
figva, ax_flata = histogram(tr.acceleration, nbins, "Acceleration ($BL/s^2$)")


# Plot curvature histograms for each fish
nbins = np.arange(-3, 3, 0.02)
figvc, ax_flatc = histogram(tr.curvature, nbins, "Curvature ($1/BL$)")
plt.show()