import cv2
import numpy as np

video_path = "test.mkv"
cap = cv2.VideoCapture(video_path)

back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=70)

def define_roi(frame):
    h, w, _ = frame.shape
    roi_top = int(h * 0.4)
    roi_bottom = int(h * 0.8)
    roi = frame[roi_top:roi_bottom, :]
    return roi, roi_top

def determine_direction(current_y, previous_y):
    if previous_y is None:
        return "unknown"
    if current_y < previous_y:
        return "away_from_camera"
    elif current_y > previous_y:
        return "towards_camera"
    return "stationary"

allowed_direction = "away_from_camera"
detections = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    roi, roi_top = define_roi(frame)
    fg_mask = back_sub.apply(roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2

            object_id = None
            for obj_id, (_, prev_y) in detections.items():
                if abs(center_y - prev_y) < 20:
                    object_id = obj_id
                    break

            if object_id is None:
                object_id = len(detections) + 1

            current_detections.append((object_id, center_y))

            previous_y = detections.get(object_id, (None, None))[1]
            movement_direction = determine_direction(center_y, previous_y)

            if movement_direction != "unknown" and movement_direction != allowed_direction:
                color = (0, 0, 255)
                text = "WRONG WAY"
            else:
                color = (0, 255, 0)
                text = "Correct Way"

            cv2.rectangle(frame, (x, y + roi_top), (x + w, y + h + roi_top), color, 2)
            cv2.putText(frame, f"ID: {object_id} {text}", (x, y + roi_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    detections = {obj_id: (center_y, center_y) for obj_id, center_y in current_detections}

    cv2.imshow("Vehicle Wrong-Way Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
