from ultralytics import DEYO

# Load a COCO-pretrained RT-DETR-l model
model = DEYO("/Users/binsusumusou/TRYO/ultralytics/cfg/models/11/TRYs.yaml").load("yolo11s.pt")

# Display model information (optional)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco128.yaml", epochs=1, freeze=23, batch=2, imgsz=640)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image