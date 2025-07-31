import cv2
import torch
import os
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort

def track_people_in_directory(input_dir, output_dir, model_name='yolov5s', confidence_threshold=0.5):
    """
    Detects and tracks people in a directory of image frames.

    Args:
        input_dir (str): Path to the directory containing input frames.
        output_dir (str): Path to the directory to save frames with tracking overlays.
        model_name (str): Name of the YOLOv5 model to use.
        confidence_threshold (float): Confidence threshold for person detection.
    """
    # Load the YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=30)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')])

    for frame_file in frame_files:
        frame_path = os.path.join(input_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Detect people in the frame
        results = model(frame)
        detections = results.pandas().xyxy[0]
        person_detections = detections[detections['name'] == 'person']

        # Format detections for DeepSort: [x1, y1, x2, y2], confidence, class
        bbs = []
        for index, row in person_detections.iterrows():
            if row['confidence'] > confidence_threshold:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                confidence = row['confidence']
                # The deep_sort_realtime library expects a list of tuples,
                # where each tuple is (bounding_box, score, class_name).
                bbs.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'person'))

        # Update the tracker with new detections
        tracks = tracker.update_tracks(bbs, frame=frame)

        # Draw bounding boxes and track IDs on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the output frame
        output_frame_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_frame_path, frame)

    print(f"Processed {len(frame_files)} frames. Results saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track people in a directory of frames.")
    parser.add_argument("input_dir", type=str, help="Directory of input frames.")
    parser.add_argument("output_dir", type=str, help="Directory to save tracked frames.")
    parser.add_argument("--model", type=str, default="yolov5s", help="YOLOv5 model name.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection.")
    args = parser.parse_args()

    track_people_in_directory(args.input_dir, args.output_dir, args.model, args.conf)
