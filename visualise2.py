import cv2
import pandas as pd
import ast

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def parse_bbox(bbox_str):
    # Remove brackets and split by spaces
    coords = bbox_str.strip('[]').split()
    return [float(coord) for coord in coords]

# Read the CSV file
results = pd.read_csv('test.csv')

# Load the video
video_path = 'sample3.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get results for current frame
    frame_results = results[results['frame_nmr'] == frame_number]

    for _, row in frame_results.iterrows():
        # Draw car bounding box
        car_bbox = parse_bbox(row['car_bbox'])
        draw_border(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (0, 255, 0), 3, 15, 10)

        # Draw license plate bounding box
        lp_bbox = parse_bbox(row['license_plate_bbox'])
        cv2.rectangle(frame, (int(lp_bbox[0]), int(lp_bbox[1])), (int(lp_bbox[2]), int(lp_bbox[3])), (0, 0, 255), 2)

        # Add license plate text
        license_plate_text = row['license_number']
        cv2.putText(frame, str(license_plate_text), (int(lp_bbox[0]), int(lp_bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Write the frame
    out.write(frame)

    frame_number += 1

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed. Output saved as 'output.mp4'")