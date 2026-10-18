"""Export MangaLens (YOLO11n-seg) to ONNX for the Go pipeline."""
from ultralytics import YOLO

PT = r"E:\work\transl\models\mangalens-bubble-seg.pt"

model = YOLO(PT)
path = model.export(format="onnx", imgsz=1600, opset=17, simplify=True, dynamic=False)
print("[OK]", path)
