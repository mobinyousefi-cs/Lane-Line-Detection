# Lane Line Detection using AI

A clean, production‑ready Python project for lane line detection with a classical computer‑vision pipeline (color + gradient thresholding, perspective transform, sliding windows, polynomial fit) and an extensible deep‑learning stub (U‑Net) for future work. Built with your preferred `src/` layout, MIT license, CI, and tests.

## ✨ Features
- Robust classical CV lane detection on images & videos (OpenCV, NumPy).
- Color/gradient thresholding with configurable parameters (YUV/HLS).
- Perspective (bird's‑eye) transform with auto‑computed or manual ROI.
- Sliding‑window search + polynomial fitting, curvature and lane offset.
- Video processing CLI with real‑time overlays.
- Extensible DL stub (`U-Net`) for future training on TuSimple/CULane.
- Reproducible project scaffold: lint (Ruff), format (Black), tests (pytest), CI (GitHub Actions).

## 🏗️ Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

## 📦 Usage
### 1) Run on image(s)
```bash
python -m lane_ai.main --image path/to/image.jpg --show
```

### 2) Run on a video
```bash
python -m lane_ai.main --video path/to/video.mp4 --out out.mp4 --display
```

### 3) Configuration
Tune thresholds and ROI in `config.yaml`.

## 🔧 Config (config.yaml)
```yaml
random_seed: 42
pipeline:
  color_space: HLS
  s_thresh: [120, 255]
  l_thresh: [200, 255]
  sobel:
    enable: true
    orient: x
    ksize: 3
    mag_thresh: [30, 255]
  roi:
    mode: auto  # auto | manual
    manual_points: [[0.15,0.95],[0.85,0.95],[0.6,0.6],[0.4,0.6]]
  perspective:
    dst_points: [[0.2,1.0],[0.8,1.0],[0.8,0.0],[0.2,0.0]]
  sliding_window:
    nwindows: 9
    margin: 100
    minpix: 50
  smoothing_frames: 5
```

## 🧪 Tests
```bash
pytest -q
```

## 📊 Datasets
For classical CV, no dataset is required. For deep learning, consider:
- TuSimple Lane Detection
- CULane

## 📜 License
MIT — © 2025 [Mobin Yousefi](https://github.com/mobinyousefi-cs)