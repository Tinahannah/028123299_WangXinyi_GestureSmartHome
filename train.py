from ultralytics import YOLO

# Load a pretrained YOLO model for pose estimation
model = YOLO("yolo11n.pt")

# # Train the model
results = model.train(data="close.yaml", epochs=100, imgsz=640)

# save trained model
model.save("total.pt")

# Save the model in ONNX format
model.export(format="onnx")