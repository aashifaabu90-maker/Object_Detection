from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.track(
    source=0,              # webcam
    show=True,             # display output
    tracker="bytetrack.yaml"
)
