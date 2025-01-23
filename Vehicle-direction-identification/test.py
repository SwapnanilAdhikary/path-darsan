# File: main.py
import cv2
from yolo import YoloDetection
from tracker import Tracker
import json

def load_model():
    model = YoloDetection(
        "C://Users//adhik//OneDrive//Desktop//Traffic Management System//vehicle Direction detection//Vehicle-direction-identification//yolov4-tiny.weights",
        "C://Users//adhik//OneDrive//Desktop//Traffic Management System//vehicle Direction detection//Vehicle-direction-identification//yolov4.cfg",
        "C://Users//adhik//OneDrive//Desktop//Traffic Management System//vehicle Direction detection//Vehicle-direction-identification//coco.names",
        416,
        416
    )
    return model

def main():
    tracker = Tracker()
    model = load_model()

    cap = cv2.VideoCapture("C://Users//adhik//OneDrive//Desktop//Traffic Management System//new traffic//3691361-hd_1920_1080_30fps.mp4")
    desired_width = 1280
    desired_height = 720

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (desired_width, desired_height))
        detections = model.process_frame(frame)
        tracking_results = tracker.track(detections)

        for result in tracking_results:
            x, y, w, h = result["points"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, result["direction"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# File: tracker.py
from vehicle_direction import find_angle_distance
from sort import Sort
import numpy as np

class Tracker:
    CLASS_NAMES = ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "person"]

    def __init__(self):
        super(Tracker, self).__init__()
        self.car_tracker = Sort()
        self.trackers_centers = {}

    def track(self, output):
        dets = []
        track_results = []

        if len(output) > 0:
            for c in output:
                if c[0] not in Tracker.CLASS_NAMES:
                    continue
                x, y, w, h = c[1:5]
                dets.append(np.array([x, y, x + w, y + h, c[5]]))

            track_bbs_ids = self.car_tracker.update(np.array(dets))

            for d in track_bbs_ids:
                d = d.astype(int)
                tid, x, y, w, h = str(d[4]), d[0], d[1], d[2] - d[0], d[3] - d[1]
                if tid not in self.trackers_centers:
                    self.trackers_centers[tid] = []
                self.trackers_centers[tid].append([x + w // 2, y + h // 2])
                direction = "None"

                if len(self.trackers_centers[tid]) >= 20:
                    direction = find_angle_distance(self.trackers_centers[tid])

                track_results.append({
                    "track_id": tid,
                    "points": [x, y, w, h],
                    "class": "vehicle",
                    "direction": direction
                })

        return track_results

# File: vehicle_direction.py
import numpy as np
from scipy.spatial import distance

def find_angle_distance(points):
    d = calculate_covered_distance(points[-20:])
    if d > 30:
        points = points[-40:]
        size = len(points) // 4
        points = points[::size]
        p1, p2, p3, p4 = points[-4:]

        if calculate_covered_distance([p2, p4]) > 20:
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p4) - np.array(p3)
            angle = np.degrees(np.arccos(np.clip(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)), -1.0, 1.0)))

            if 0 <= angle <= 23:
                return "straight"
            elif 23 < angle < 90:
                diff = (p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])
                return "right" if diff > 0 else "left" if diff < 0 else "straight"
        return "None"
    return "stopped"

def calculate_covered_distance(points):
    return sum(distance.euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))

# File: yolo.py
import cv2
import numpy as np

class YoloDetection:
    def __init__(self, model_path, config, classes, width, height, scale=0.00392, thr=0.5, nms=0.4):
        super(YoloDetection, self).__init__()
        self.__confThreshold = thr
        self.__nmsThreshold = nms
        self.__scale = scale
        self.__width = width
        self.__height = height

        self.__net = cv2.dnn.readNet(model_path, config)
        self.__classes = self.__load_classes(classes)

    def __load_classes(self, classes_path):
        with open(classes_path, 'rt') as f:
            return f.read().strip().split("\n")

    def get_output_layers_name(self):
        return [self.__net.getLayerNames()[i - 1] for i in self.__net.getUnconnectedOutLayers()]

    def post_process_output(self, frame, outs):
        frame_height, frame_width = frame.shape[:2]
        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.__confThreshold:
                    center_x, center_y = int(detection[0] * frame_width), int(detection[1] * frame_height)
                    width, height = int(detection[2] * frame_width), int(detection[3] * frame_height)
                    left, top = center_x - width // 2, center_y - height // 2

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.__confThreshold, self.__nmsThreshold)
        return indices, boxes, confidences, class_ids

    def process_frame(self, frame):
        blob = cv2.dnn.blobFromImage(frame, self.__scale, (self.__width, self.__height), [0, 0, 0], True, crop=False)
        self.__net.setInput(blob)
        outs = self.__net.forward(self.get_output_layers_name())

        indices, boxes, confidences, class_ids = self.post_process_output(frame, outs)
        detected_objects = []

        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box
            detected_objects.append([
                self.__classes[class_ids[i]], int(left), int(top), int(width), int(height), confidences[i]
            ])

        return detected_objects
