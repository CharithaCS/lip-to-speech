import os
import glob
import cv2
import yaml
import argparse
import numpy as np
import torch
import face_alignment
import ffmpeg
from tqdm import tqdm
from scipy.signal import savgol_filter

def smooth_landmarks(landmarks, window_length=5, polyorder=2):
    """
    Applies a Savitzky-Golay filter to smooth the landmark coordinates.
    This reduces jitter between frames.
    """
    smoothed_landmarks = np.zeros_like(landmarks)
    for i in range(landmarks.shape[1]):
        for j in range(landmarks.shape[2]):
            # The window length must be odd and less than the number of points.
            actual_window_length = min(window_length, len(landmarks[:, i, j]))
            if actual_window_length % 2 == 0:
                actual_window_length -= 1
            
            if actual_window_length > polyorder:
                smoothed_landmarks[:, i, j] = savgol_filter(landmarks[:, i, j], actual_window_length, polyorder)
            else:
                # Not enough points to apply the filter, use original data
                smoothed_landmarks[:, i, j] = landmarks[:, i, j]
    return smoothed_landmarks

def process_videos(config):
    """
    Main function to align faces and extract audio.
    Reads from the raw frames directory and the original corpus.
    Writes to the final aligned directory.
    """
    raw_frames_base_dir = config['path']['raw_frames_path']
    final_output_base_dir = config['path']['preprocessed_path']
    original_grid_dir = config['path']['corpus_path']
    device = config['preprocessing']['device']
    
    print(f"Initializing face alignment model on device: {device}")
    # Fixed: Changed L2D to TWO_D
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)

    speaker_dirs = sorted(glob.glob(os.path.join(raw_frames_base_dir, "s*")))
    if not speaker_dirs:
        print(f"Error: No speaker directories found in {raw_frames_base_dir}. Please run the first script.")
        return
    print(f"Found {len(speaker_dirs)} speaker directories in the raw frames folder.")

    for speaker_dir in tqdm(speaker_dirs, desc="Finalizing Speakers"):
        video_dirs = sorted(glob.glob(os.path.join(speaker_dir, "*")))
        
        for video_frame_dir in tqdm(video_dirs, desc=f"Videos for {os.path.basename(speaker_dir)}", leave=False):
            try:
                video_id = os.path.basename(video_frame_dir)
                speaker_id = os.path.basename(os.path.dirname(video_frame_dir))
                print(f"Processing {speaker_id}/{video_id}")

                # --- 1. Define Output Paths ---
                final_video_dir = os.path.join(final_output_base_dir, speaker_id, video_id)
                final_frames_dir = os.path.join(final_video_dir, 'frames')
                audio_output_path = os.path.join(final_video_dir, 'audio.wav')

                if os.path.exists(final_frames_dir) and os.path.exists(audio_output_path):
                    continue
                
                os.makedirs(final_frames_dir, exist_ok=True)

                # --- 2. Find Original Video for Audio Extraction ---
                original_video_path = os.path.join(original_grid_dir, speaker_id, "video", f"{video_id}.mpg")
                if not os.path.exists(original_video_path):
                    print(f"Warning: Original video not found at {original_video_path}. Skipping.")
                    continue

                # --- 3. Extract Audio ---
                if not os.path.exists(audio_output_path):
                    ffmpeg.input(original_video_path).output(
                        audio_output_path,
                        acodec='pcm_s16le', ac=1, ar='16000'
                    ).run(quiet=True, overwrite_output=True)

                # --- 4. Load Raw Frames ---
                frame_files = sorted(glob.glob(os.path.join(video_frame_dir, "*.jpg")))
                if not frame_files: continue
                
                raw_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in frame_files]
                
                # --- 5. Landmark Detection ---
                print(f"Processing {len(raw_frames)} frames for landmark detection")
                all_landmarks = []
                
                for i, frame in enumerate(raw_frames):
                    try:
                        # Use get_landmarks_from_image instead of batch processing
                        landmarks = fa.get_landmarks_from_image(frame)
                        all_landmarks.append(landmarks)
                        if i % 25 == 0:  # Progress indicator
                            print(f"Processed {i+1}/{len(raw_frames)} frames")
                    except Exception as e:
                        print(f"Error in landmark detection for frame {i}: {e}")
                        all_landmarks.append(None)
                
                # --- 6. Handle Missing Detections & Stabilize ---
                print(f"Got landmarks for {len(all_landmarks)} frames")
                if not any(all_landmarks):
                    print(f"Warning: No faces detected in {video_frame_dir}. Skipping.")
                    continue

                # Fill in None landmarks
                for i in range(len(all_landmarks)):
                    if all_landmarks[i] is None: 
                        all_landmarks[i] = all_landmarks[i-1] if i > 0 and all_landmarks[i-1] is not None else None
                        
                if all_landmarks[0] is None:
                    first_valid = next((item for item in all_landmarks if item is not None), None)
                    if first_valid is None: 
                        print(f"Warning: No valid landmarks found for {video_frame_dir}. Skipping.")
                        continue
                    for i in range(len(all_landmarks)):
                        if all_landmarks[i] is None: 
                            all_landmarks[i] = first_valid
                        else: 
                            break
                
                # Convert to numpy array - be careful here
                print("Converting landmarks to numpy array")
                try:
                    # Check if we have valid landmarks
                    valid_landmarks = [l for l in all_landmarks if l is not None]
                    if not valid_landmarks:
                        print(f"Warning: No valid landmarks after processing for {video_frame_dir}. Skipping.")
                        continue
                    
                    # Take the first landmark from each frame (assuming single face)
                    landmarks_np = np.array([l[0] if isinstance(l, list) and len(l) > 0 else l for l in all_landmarks])
                    print(f"Landmarks array shape: {landmarks_np.shape}")
                    
                    smoothed_landmarks = smooth_landmarks(landmarks_np)
                    print("Landmarks smoothed successfully")
                except Exception as e:
                    print(f"Error processing landmarks: {e}")
                    continue

                # --- 7. Crop, Resize, and Save Face Frames ---
                # Extract config values once to avoid repeated dictionary access
                crop_scale = config['preprocessing']['crop_scale']
                frame_size = config['preprocessing']['frame_size']
                
                for i, (frame, landmarks) in enumerate(zip(raw_frames, smoothed_landmarks)):
                    min_xy = np.min(landmarks, axis=0)
                    max_xy = np.max(landmarks, axis=0)
                    center = (min_xy + max_xy) / 2
                    size = np.max(max_xy - min_xy) * crop_scale
                    half_size = size / 2
                    x1, y1 = int(center[0] - half_size), int(center[1] - half_size)
                    x2, y2 = int(center[0] + half_size), int(center[1] + half_size)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cropped_frame = frame[y1:y2, x1:x2]
                    
                    # Check if cropped frame is valid
                    if cropped_frame.size == 0:
                        print(f"Warning: Empty cropped frame for {video_frame_dir}, frame {i}")
                        continue
                    
                    resized_frame = cv2.resize(cropped_frame, (frame_size, frame_size))
                    cv2.imwrite(os.path.join(final_frames_dir, f"{i:04d}.jpg"), 
                                cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"An error occurred while processing {video_frame_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    # Load configuration from YAML file
    config_path = 'configs/preprocess_config.yaml'
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run the final processing pipeline
    print("--- Starting Step 2: Final Audio Extraction & Face Alignment ---")
    process_videos(config)
    print("--- Step 2 Complete ---")
    print("\n\n!! PREPROCESSING FINISHED !!")
    print(f"Your final, self-contained dataset is located at: {config['path']['preprocessed_path']}")
    print("You can now safely delete the original GRID dataset and the temporary raw frames directory.")