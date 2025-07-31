import os
import argparse
import datetime
from Preprocessing.Preprocessing import extract_frames
from Tracking.tracking import track_people_in_directory
# from Feature_Extraction.feature_extractor import CNNFeatureExtractor

def main(args):
    """
    Main pipeline to run the pedestrian tracking system.
    """
    # --- Setup ---
    # Create a unique output directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    frames_dir = os.path.join(run_output_dir, "extracted_frames")
    tracked_frames_dir = os.path.join(run_output_dir, "tracked_frames")

    print(f"Starting pipeline. Results will be saved in: {run_output_dir}")

    # --- Step 1: Preprocessing ---
    # Extract frames from the input video
    print("\nStep 1: Extracting frames from video...")
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return
    extract_frames(args.video_path, frames_dir)
    print("Frame extraction complete.")

    # --- Step 2: Person Detection and Tracking ---
    print("\nStep 2: Running person detection and tracking...")
    track_people_in_directory(frames_dir, tracked_frames_dir, model_name=args.model, confidence_threshold=args.conf)
    print("Tracking complete.")

    # --- Step 3: Feature Extraction (Placeholder) ---
    print("\nStep 3: Feature Extraction (Placeholder)...")
    # In a full implementation, you would now iterate through the tracking results.
    # For each tracked person (each track_id), you would crop their bounding box
    # from each frame they appear in and extract features using the CNNFeatureExtractor.

    # Example placeholder logic:
    # feature_extractor = CNNFeatureExtractor()
    # all_features = {} # Dict to store features, e.g., {track_id: [feature1, feature2, ...]}
    # For each track in the tracker's output:
    #   crop image based on bounding box
    #   features = feature_extractor.extract_features(cropped_image)
    #   if track.track_id not in all_features:
    #     all_features[track.track_id] = []
    #   all_features[track.track_id].append(features)
    print("This step would involve extracting visual features for each tracked person.")

    # --- Step 4: Person Re-identification (Placeholder) ---
    print("\nStep 4: Person Re-identification (Placeholder)...")
    # With the extracted features, you could now implement re-identification logic.
    # For example, if a track is lost and a new track appears, you could compare
    # the features of the new track with those of previously lost tracks to see
    # if it's the same person reappearing.
    print("This step would use the extracted features to re-identify people across camera views or after occlusions.")

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Tracking Pipeline.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Parent directory to save the results.")

    # Tracking arguments
    parser.add_argument("--model", type=str, default="yolov5s", help="YOLOv5 model name for tracking.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for person detection in tracking.")

    args = parser.parse_args()
    main(args)
