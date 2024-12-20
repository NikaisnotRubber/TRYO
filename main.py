from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("/Users/binsusumusou/TRYO/ultralytics/cfg/models/v8/TRY8.yaml").load("yolo11s.pt")

# Display model information (optional)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=5, imgsz=640, device = 'mps', freeze=23)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image