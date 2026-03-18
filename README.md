# Depth Estimation Tool

A Streamlit web app that estimates real-world distances to animals in camera-trap images. It uses the [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2) AI model to generate a depth map from a single image, then lets you calibrate that depth map with known distances so every detected animal gets a distance estimate.

This is a standalone development tool that will eventually be incorporated into the [AddaxAI](https://github.com/PetervanLunteren/AddaxAI) Streamlit app.

![Example depth map](assets/example.png)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/PetervanLunteren/st-depth-estimation.git
cd st-depth-estimation
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch (`torch`) is required. If you need GPU support or a specific CUDA version, install PyTorch separately first following the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining requirements.

## Usage

Run the app:

```bash
streamlit run app.py
```

This opens the tool in your browser (typically `http://localhost:8501`).

## Input data

The app expects:

- **An image folder** (`test-imgs/` by default) containing camera-trap JPEGs.
- **A detections CSV** (`test-imgs/results_detections.csv`) with columns for each image's detected animals, their bounding boxes, confidence scores, and metadata. This CSV is produced by an upstream AI recognition model (e.g. via AddaxAI).

<details>
<summary>Example detections CSV</summary>

| relative_path | label | confidence | bbox_left | bbox_top | bbox_right | bbox_bottom | DateTimeOriginal | Latitude | Longitude |
|---|---|---|---|---|---|---|---|---|---|
| img_0001.jpg | species Alcelaphus buselaphus | 0.92241 | 558 | 398 | 941 | 654 | 18/01/2013 08:58 | 0.278 | 36.874 |
| img_0001.jpg | family Bovidae | 0.94621 | 1058 | 468 | 1227 | 574 | 18/01/2013 08:58 | 0.278 | 36.874 |
| img_0002.jpg | species Alcelaphus buselaphus | 0.97218 | 552 | 397 | 919 | 652 | 18/01/2013 08:58 | 0.278 | 36.874 |

</details>

## How to operate the tool

The sidebar guides you through four numbered steps:

### Step 1: Select an Image

Pick a representative camera-trap image from the deployment. This is the image the AI will analyse for depth and that you will use to calibrate distances.

### Step 2: Generate Depth Map

Choose a model size (Small / Base / Large) and click **Generate Depth Map**. The AI model estimates how far each pixel is from the camera and produces a colour-coded depth map:

- **Purple/blue** = closer to the camera
- **Yellow** = farther from the camera

Once generated, the preview shows an interactive slider you can drag left/right to compare the original photo with the depth map.

### Step 3: Calibrate Distances

The depth map only gives *relative* depth (closer vs. farther), not real-world distances. To convert depth into metres you need to provide reference points -- objects in the image whose distance from the camera you know.

1. **Click** on the preview image (or expand the clickable image below the slider) to mark a point.
2. **Enter** the known distance in metres in the sidebar.
3. Click **Add calibration point**.
4. Repeat for at least 2 points at different distances (e.g. a tree at 5 m and a post at 15 m). More points at varied distances improve accuracy.

The **Calibration Quality** section shows an accuracy score (R²). Values above 0.9 are good; below 0.8 means you should add more or better-spread reference points.

Calibration points can be saved to disk and reloaded later.

### Step 4: Compute Distances

Click **Compute distances for all detections** to apply the calibration across every detection in the CSV. The detections table at the bottom updates with the first 10 results as a preview, and you can download the full results as a CSV.

## Display options

- **Show bounding boxes / sampling points** -- toggle overlays on the preview image.
- **Sampling point position** -- choose where on each animal's bounding box to measure depth: `bottom-center` (at the animal's feet, usually best) or `center` (middle of the box).
- **Distance unit** -- switch between metres and feet for the table and CSV export.
- **Date format** -- choose DD/MM/YYYY, MM/DD/YYYY, or YYYY-MM-DD.
- **Filter by label** -- show only detections matching a specific species or category.

## Project structure

```
st-depth-estimation/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md
├── assets/
│   └── example.png                 # Screenshot for README
└── test-imgs/                      # Example deployment
    ├── results_detections.csv      # Detection results with bounding boxes
    ├── calibration_points.csv      # Saved calibration points (created at runtime)
    └── img_0001.jpg ... img_0019.jpg
```

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation and CSV I/O |
| `Pillow` | Image drawing (overlays, markers) |
| `numpy` | Numerical computation (calibration fitting) |
| `torch` | Deep learning runtime |
| `transformers` (>=4.47.1) | HuggingFace model loading (DepthAnything V2) |
| `streamlit-image-coordinates` | Click detection on images |
| `streamlit-image-comparison` | Side-by-side image comparison slider |

## TODO

- [ ] Support user-uploaded images and CSV files instead of requiring a fixed folder
- [ ] Compute depth maps per-image so calibration applies correctly across images with different camera positions
- [ ] Add animal height/size estimation using bounding box dimensions and computed distance
- [ ] Explore alternative sampling strategies for oddly-shaped animals (snakes, giraffes) where bottom-center may not be ideal
- [ ] Add a calibration visualization (plot of reference points vs. fitted curve)
- [ ] Batch processing progress bar for large deployments
- [ ] Integrate into the main AddaxAI Streamlit app
