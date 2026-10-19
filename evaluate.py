import sys
from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
model = YOLO(weights)

metrics = model.val(data="data.yaml", split="val")

print("\n===== RESULTS =====")
print(f"mAP50      : {metrics.box.map50 * 100:.1f} %")
print(f"mAP50-95   : {metrics.box.map * 100:.1f} %")
print(f"Precision  : {metrics.box.mp * 100:.1f} %")
print(f"Recall     : {metrics.box.mr * 100:.1f} %")

print("\nAP50 per class:")
for class_id, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
    print(f"  {model.names[int(class_id)]:<12} {ap * 100:.1f} %")
