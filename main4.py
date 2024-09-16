from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort import Sort
from util import get_car, read_license_plate, write_csv
import os

results = {}

mot_tracker = Sort()

# load models with lower confidence threshold
coco_model = YOLO('yolov8n.pt')
coco_model.conf = 0.25  # lower confidence threshold
license_plate_detector = YOLO('license_plate_detector.pt')
license_plate_detector.conf = 0.25  # lower confidence threshold

# load video
cap = cv2.VideoCapture('./sample3.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        print(f"Frame {frame_nmr}: Detected {len(detections_)} vehicles")

        # track vehicles
        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.array(detections_))
        else:
            track_ids = np.empty((0, 5))
        
        print(f"Frame {frame_nmr}: Tracking {len(track_ids)} vehicles")

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        print(f"Frame {frame_nmr}: Detected {len(license_plates)} license plates")

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                    print(f"Frame {frame_nmr}: Detected license plate {license_plate_text} for car {car_id}")

# write results
output_path = os.path.abspath('./test.csv')
try:
    write_csv(results, output_path)
    print(f"Results written to {output_path}")
except PermissionError:
    print(f"Permission denied when writing to {output_path}. Please ensure you have write access to this location.")
    alternative_path = os.path.join(os.path.expanduser('~'), 'anpr_results.csv')
    try:
        write_csv(results, alternative_path)
        print(f"Results written to alternative location: {alternative_path}")
    except Exception as e:
        print(f"Failed to write results to alternative location. Error: {e}")

print(f"Total frames processed: {frame_nmr}")
print(f"Total results: {sum(len(frame_results) for frame_results in results.values())}")