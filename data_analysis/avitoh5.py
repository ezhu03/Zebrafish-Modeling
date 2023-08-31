import cv2
import h5py
import numpy as np

# Open the AVI video file
avi_filename = 'data_analysis\\individual_1.avi'
cap = cv2.VideoCapture(avi_filename)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an HDF5 file
h5_filename = 'data_analysis\\output_1fishgroup16minacc_0823.h5'
h5_file = h5py.File(h5_filename, 'w')

# Create a dataset to store frames
frame_dataset = h5_file.create_dataset('frames', shape=(frame_count, frame_height, frame_width, 3), dtype=np.uint8)

# Process frames and save to HDF5
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save the frame to HDF5 dataset
    frame_dataset[frame_index] = frame
    
    frame_index += 1

# Close the HDF5 file and video capture
h5_file.close()
cap.release()
