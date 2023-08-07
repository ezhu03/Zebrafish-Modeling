import cv2
import h5py
import os

# Replace 'input.avi' with the path to your AVI file
input_video_path = 'singlefishtest_2023-07-31-152515-0000.avi'

# Replace 'output_folder' with the path to the folder where you want to save the HDF5 files
output_folder = 'data_output/singlefishtest_2023-07-31-152515-0000'

# Replace 'output_file_name.h5' with the desired name for the output HDF5 file
output_file_name = 'singlefishtest_2023-07-31-152515-0000.h5'

# Initialize VideoCapture object to read frames from the AVI file
video_capture = cv2.VideoCapture(input_video_path)

# Create the HDF5 file
with h5py.File(output_file_name, 'w') as h5file:
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Create a dataset within the HDF5 file to store the frame
        h5file.create_dataset(f'/frame_{frame_count:04d}', data=frame)

        frame_count += 1

# Release the VideoCapture object
video_capture.release()