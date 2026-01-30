# Low Resolution Estimation (ML Portfolio)

A robust machine learning tool designed to estimate the **true resolution** and **quality score** of images. This model detects upscaled images (whether bicubic, bilinear, or nearest-neighbor) and predicts their effective resolution, distinguishing between native sharp details and interpolated pixels.

![Demo](https://img.shields.io/badge/Demo-Gradio-orange) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

## Key Features

- **Robust Quality Scoring**: Predicts a score from `0.0` (pixelated/blurry) to `1.0` (sharp/native).
    - **Advanced Augmentation**: Trained with miscellaneous degradation types (**Bicubic, Bilinear, Nearest-Neighbor, Lanczos**, etc.) to handle real-world upscaling artifacts, including pixelation and blur.
- **True Resolution Estimation**: Infers the effective `Width x Height` of the original source before upscaling.
- **Region-Aware Inference**: Uses a **5-crop strategy** (Center + 4 Corners) to analyze the entire image surface for consistent scoring.
- **Lightweight Architecture**: built on `MobileNetV3-Small` for millisecond-latency inference on CPU or GPU.

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/low-res-estimation.git
cd low-res-estimation
pip install -r requirements.txt
```

### 2. Interactive Demo (Gradio)

Launch the web interface to test images interactively:

```bash
python gradio_app.py
```

Open your browser at `http://localhost:7860`.
- **Upload**: Any image (upscaled, blurry, or sharp).
- **View**: The "Quality Score" and "Estimated True Resolution".

### 3. Command Line Inference

Estimate resolution for a specific file or directory:

```bash
python test.py --input path/to/image.jpg --model_path checkpoints/best_model.pth
```

**Output:**
```text
Image: test.jpg
  Current Size: 1920x1080
  Quality Score: 0.4512
  Estimated True Size: 866x487
```

## Training

To train the model on your own dataset (requires a folder of high-res images):

```bash
python train.py --data_dir /path/to/high-res-images --epochs 50 --save_dir checkpoints
```

**Training Features:**
- **Automatic Degradation**: The script automatically generates low-res pairs on-the-fly using random scalings and interpolation filters.
- **Resumable**: Use `--resume checkpoints/last.pth` to continue training.

## Verification

To verify the model's robustness against random degradations:

```bash
python random_test.py --input /path/to/images
```

This script will artificially degrade images and measure the model's prediction error (MAE), ensuring it remains accurate across different quality levels.

## License

MIT License
