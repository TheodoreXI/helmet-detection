import os
import sys
from ultralytics import YOLO

if len(sys.argv) < 2:
    sys.exit("Usage: python detect.py <image | video | folder | 0 for webcam>")

source = sys.argv[1]
is_webcam = source.isdigit()
if is_webcam:
    source = int(source)

model = YOLO("best.pt")

results = model.predict(
    source=source,
    conf=0.25,
    save=True,
    show=is_webcam,
    stream=True,
    project=os.path.abspath("outputs"),
    name="predictions",
    exist_ok=True,
)

for r in results:
    found = [r.names[int(c)] for c in r.boxes.cls]
    if not is_webcam:
        print(r.path, "->", found if found else "nothing detected")

print("Done. Results saved in outputs/predictions/")
