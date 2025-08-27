import ffmpeg
import cv2
import os

def extract_frame(input_path: str, frame_number: int, output_path: str) -> None:
    """
    Extract a specific frame from an MP4 video file and save it as a .png in the Downloads folder.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        raise ValueError(f"Frame number {frame_number} is out of range (0-{total_frames-1})")

    # Position and read the requested frame without any resizing or color conversions beyond default BGR
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_number}")

    # Derive an output stem so we can save multiple formats
    stem, _ = os.path.splitext(output_path)
    png_path = f"{stem}.png"
    tiff_path = f"{stem}.tiff"

    # Save PNG with no compression (lossless, maximum fidelity to decoded frame)
    cv2.imwrite(png_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # Save TIFF as an additional lossless option (some scientific tools prefer TIFF)
    # Try explicitly setting no compression when supported; fall back to default otherwise.
    try:
        cv2.imwrite(tiff_path, frame, [cv2.IMWRITE_TIFF_COMPRESSION, 1])  # 1 == no compression
    except Exception:
        cv2.imwrite(tiff_path, frame)

    cap.release()
    print(f"Frame {frame_number} saved to {png_path} and {tiff_path}")

if __name__ == "__main__":
    input_path = "data_analysis/trajectorytools/1fish/reflectionvisualization/reflection-visualization-half-21dpf-1.mp4"
    
    # Ask user for frame number
    frame_number = int(input("Enter the frame number to extract: "))
    downloads_folder = os.path.expanduser("~/Downloads")
    output_frame_path = os.path.join(downloads_folder, f"extracted_frame_{frame_number}")
    
    # Extract and save the frame
    extract_frame(input_path, frame_number, output_frame_path)