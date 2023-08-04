import cv2
import h5py
import os

# Replace 'input.avi' with the path to your AVI file
input_video_path = 'singlefishtest_2023-07-31-152515-0000.avi'

# Initialize VideoCapture object to read frames from the AVI file
video_capture = cv2.VideoCapture(input_video_path)

# Replace 'output_folder' with the path to the folder where you want to save the HDF5 files
output_folder = 'data_output/singlefishtest_2023-07-31-152515-0000'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through the video and write each frame to a separate HDF5 file
frame_count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1

    # Replace 'output_file_prefix' with the desired prefix for the HDF5 file names
    output_file_name = os.path.join(output_folder, f'output_file_prefix_{frame_count:04d}.h5')

    # Write the frame data to the HDF5 file
    with h5py.File(output_file_name, 'w') as h5file:
        h5file.create_dataset('/frame', data=frame)

# Release the VideoCapture object
video_capture.release()
