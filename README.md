# PixelShift — Pixel-Drift Reconstruction

Learn a **displacement field** that rearranges every pixel from a source image to reconstruct a target image. No new pixels are created — only existing ones are shifted.

## How It Works

1. **Initialize** an identity flow grid (every pixel maps to itself)
2. **Warp** the source image using the current flow grid
3. **Compare** the warped result to the target using three losses:
   - **Multi-Scale Sinkhorn / SWD** — matches global color distributions from coarse-to-fine pyramid levels
   - **Perceptual (VGG19)** — matches structural features
   - **Total Variation** — keeps displacement smooth and fluid
4. **Backpropagate** error directly into the flow grid values
5. **Repeat** — pixels gradually drift into the target shape

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python main.py
```

1. Click **Browse Source** → load your source image (the "pixel bag")
2. Click **Browse Target** → load the target image (the shape to recreate)
3. Adjust parameters if desired
4. Click **▶ Start** — watch pixels drift in real-time
5. Click **💾 Save** when satisfied

## Project Structure

```
PixelShift/
├── main.py                    # Entry point
├── core/
│   ├── device_manager.py      # Auto GPU detection & dual-GPU split
│   ├── flow_field.py          # Identity grid + learnable displacement
│   ├── feature_extractor.py   # Frozen VGG19 multi-layer features
│   ├── losses.py              # Sinkhorn/SWD + Perceptual + TV losses
│   └── optimizer_engine.py    # Main optimization loop
├── gui/
│   ├── main_window.py         # PyQt5 main window (dark theme)
│   ├── image_panel.py         # Image display widget
│   ├── controls_panel.py      # Parameters, buttons, loss chart
│   └── worker.py              # QThread optimization worker
├── utils/
│   └── image_utils.py         # Image I/O & tensor conversions
└── requirements.txt
```

## GPU Support

| GPUs | Feature Extractor | Optimizer |
|:----:|:-----------------:|:---------:|
| 0    | CPU               | CPU       |
| 1    | cuda:0            | cuda:0    |
| 2+   | cuda:0            | cuda:1    |

## Requirements

- Python 3.8+
- PyTorch ≥ 2.0
- PyQt5 ≥ 5.15
- torchvision, Pillow, matplotlib, numpy
- geomloss (optional — falls back to Sliced Wasserstein Distance)
