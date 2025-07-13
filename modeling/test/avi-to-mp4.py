import ffmpeg

def avi_to_mp4_ffmpeg(input_path: str, output_path: str) -> None:
    """
    Convert an AVI file to MP4 using the ffmpeg-python wrapper.

    Parameters:
        input_path (str): Path to the source .avi file.
        output_path (str): Path to the desired .mp4 output.
    """
    (
        ffmpeg
        .input(input_path)
        .output(output_path,
                vcodec="libx264",       # H.264 for video
                acodec="aac",           # AAC for audio
                strict="-2",            # allow experimental AAC if needed
                movflags="faststart"    # optimize for web streaming
        )
        .run(cmd=["/Users/ezhu/Documents/GitHub/Zebrafish-Modeling/ffmpeg"],overwrite_output=True)
    )

if __name__ == "__main__":
    avi_to_mp4_ffmpeg("/Volumes/Hamilton/Zebrafish/AVI/07.02.24/1fish-1fps-15min-7dpf-clear1_2024-07-02-143006-0000.avi", "/Users/ezhu/Downloads/output_video.mp4")