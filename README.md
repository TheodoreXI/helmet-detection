# Helmet Detection with YOLOv8

Detects whether people wear a safety helmet in images, videos or a webcam stream.
Built with Python, YOLOv8 (Ultralytics) and OpenCV (installed with Ultralytics).

## Files

| File | What it does |
|------|--------------|
| `data.yaml` | Dataset paths and class names |
| `train.py` | Fine-tunes YOLOv8n on the helmet dataset, saves `best.pt` |
| `evaluate.py` | Computes mAP50, precision and recall on the validation set |
| `detect.py` | Runs the trained model on an image, video, folder or webcam |

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## Dataset

Put the dataset in a `dataset/` folder with this structure (standard YOLO format):

```
dataset/
  train/images  train/labels
  valid/images  valid/labels
```

Then make sure the class names in `data.yaml` match the dataset.

## Usage

```
python train.py                 # train, creates best.pt
python evaluate.py              # prints mAP50 on the validation set
python detect.py photo.jpg      # detect on one image
python detect.py video.mp4      # detect on a video
python detect.py 0              # webcam (press q to quit)
```

Results of `detect.py` are saved in `outputs/predictions/`.

## Results

Validation mAP50: **XX %**

## How it works (short)

1. YOLOv8n starts already trained on COCO (general objects).
2. `train.py` fine-tunes it on helmet images so it learns the helmet classes.
3. mAP50 = how well the predicted boxes overlap the real boxes (IoU >= 0.5), averaged over classes.
