from ultralytics import DEYO

# Load a COCO-pretrained RT-DETR-l model
model = DEYO("/Users/binsusumusou/ultralytics/ultralytics/cfg/models/11/TRYs.yaml").load("yolo11s.pt")

# Display model information (optional)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=3, imgsz=640, freeze=23)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image