import os
import shutil
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project=os.path.abspath("runs"),
    name="helmet",
    exist_ok=True,
)

shutil.copy(model.trainer.best, "best.pt")
print("Training finished. Best weights saved to best.pt")
