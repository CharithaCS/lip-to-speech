import os
import glob
import cv2
import yaml
from tqdm import tqdm

def extract_frames(config):
    """
    Extracts frames from all .mpg videos in the GRID corpus.
    """
    grid_dir = config['path']['corpus_path']
    output_dir_base = config['path']['preprocessed_path']
    
    # Structure of GRID: grid_dir/speaker/video/*.mpg
    video_files = glob.glob(os.path.join(grid_dir, "*", "*", "*.mpg"))
    
    if not video_files:
        print(f"Error: No .mpg video files found in {grid_dir}. Please check the 'corpus_path' in your config file.")
        return
        
    print(f"Found {len(video_files)} video files to process.")

    for video_path in tqdm(video_files, desc="Extracting Frames"):
        try:
            # --- 1. Define Output Path ---
            # Create a unique path based on the speaker and video name
            relative_path = os.path.relpath(video_path, grid_dir)
            speaker = os.path.basename(os.path.dirname(os.path.dirname(relative_path)))
            video_id = os.path.splitext(os.path.basename(relative_path))[0]
            
            frame_output_dir = os.path.join(output_dir_base, speaker, video_id)

            # --- 2. Skip if Already Processed ---
            # The GRID corpus has 75 frames per video.
            if os.path.exists(frame_output_dir) and len(os.listdir(frame_output_dir)) >= 75:
                continue

            os.makedirs(frame_output_dir, exist_ok=True)

            # --- 3. Read Video and Save Frames ---
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_path}")
                continue
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame as a high-quality JPG
                cv2.imwrite(os.path.join(frame_output_dir, f"{frame_count:04d}.jpg"), frame)
                frame_count += 1
            
            cap.release()

        except Exception as e:
            print(f"An error occurred while processing {video_path}: {e}")
            continue

if __name__ == "__main__":
    # Load configuration from YAML file
    config_path = 'configs/preprocess_config.yaml'
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run the frame extraction pipeline
    print("--- Starting Frame Extraction ---")
    extract_frames(config)
    print("--- Frame Extraction Complete ---")