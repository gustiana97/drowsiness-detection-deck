# Driver Drowsiness Detection (YOLO)

Real-time driver drowsiness detection using a YOLO object-detection model.
Detects drowsy indicators (closed eyes / yawning) and raises an alert when the
driver appears drowsy for several consecutive frames.

Based on the project proposal: *SI5101701 Driver Drowsiness Detection Using Deep Learning*.

## Hardware / environment

- Windows 11, NVIDIA RTX 5050 Laptop (8 GB), CUDA 12.8 build of PyTorch.
- Python 3.12 virtual environment at `.venv` (PyTorch has no 3.14 wheels yet).

## Setup

```powershell
# from the project root
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Dataset

A YOLO-format **detection** dataset (images + bounding-box labels + `data.yaml`)
goes under `datasets/`. Typical classes: open eyes / closed eyes / yawn.

Downloaded from Kaggle using the Kaggle API (place `kaggle.json` in
`%USERPROFILE%\.kaggle\`).

## Train

```powershell
python scripts/train.py --data datasets/<name>/data.yaml --epochs 50 --model yolov8n.pt --device 0
```

Best weights are written to `runs/detect/drowsiness/weights/best.pt`.

## Predict on an image / folder / video

```powershell
python scripts/predict.py --weights runs/detect/drowsiness/weights/best.pt --source path/to/input
```

## Real-time webcam demo

```powershell
python scripts/webcam.py --weights runs/detect/drowsiness/weights/best.pt --camera 0
```

Press `q` to quit. A red border + beep fires when drowsiness is sustained.

Tips:
- Aim the camera at your face (tilt the laptop lid so the webcam points at you).
- Eyes behind glasses can be missed. The default `--conf 0.25` is already low;
  lower further (e.g. `--conf 0.15`) or train more epochs for better recall.

## Grad-CAM visualization

Shows which facial regions drive the prediction (eyes / mouth). Default uses a
class-targeted Grad-CAM on the high-resolution P3 head (layer 15).

```powershell
python scripts/gradcam.py --weights runs/detect/drowsiness/weights/best.pt --source datasets/drowsiness/test/images --max 8
```

Output (left: detection, right: heatmap) is saved under `runs/gradcam/`.
Use `--method eigencam` for the gradient-free variant.
