import torch
import torchvision.transforms as T
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time  # To measure FPS

# Paths
model_path = "C://Users//adhik//OneDrive//Desktop//Traffic Management System//traffic faster RCNN//fasterrcnn_resnet50_epoch_1.pth"
video_path = "C://Users//adhik//OneDrive//Desktop//Traffic Management System//traffic faster RCNN//853908-hd_1920_1080_25fps.mp4"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=False)

# Modify the predictor to match your number of classes
num_classes = 3  # Update this to the number of classes in your dataset (including background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the state dict
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# Video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Transformation for input frames
transform = T.Compose([T.ToTensor()])

# Resize dimensions for processing
resize_width, resize_height = 640, 640

# To calculate FPS
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for uniformity and better processing
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    
    # Convert frame to tensor and send to device
    input_tensor = transform(resized_frame).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Parse the detections
    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score > 0.5:  # Confidence threshold
            # Extract box coordinates
            x1, y1, x2, y2 = box.int().cpu().numpy()
            # Draw bounding boxes and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Measure FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Add FPS info on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow("Faster R-CNN Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
