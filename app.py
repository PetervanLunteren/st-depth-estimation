import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_comparison import image_comparison
from streamlit_image_coordinates import streamlit_image_coordinates
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


st.set_page_config(page_title="Depth Estimation Tool", layout="wide")
st.title("Depth Estimation Tool")
st.markdown(
    "Estimate real-world distances to animals in camera-trap images using "
    "AI depth maps and manual calibration."
)
st.info(
    "**Quick start:** Follow the numbered steps in the sidebar from top to bottom: "
    "select an image, generate a depth map, calibrate with known distances, "
    "then compute distances for all detections."
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "test-imgs")
CSV_PATH = os.path.join(IMAGE_FOLDER, "results_detections.csv")
CALIBRATION_CSV_PATH = os.path.join(IMAGE_FOLDER, "calibration_points.csv")

# DepthAnything V2 model choices
DEPTH_MODELS = {
    "Small (fastest)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large (most accurate)": "depth-anything/Depth-Anything-V2-Large-hf",
}
DEFAULT_MODEL = "Small (fastest)"


# ---------- helpers ----------
def _safe_int(x) -> Optional[int]:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


DATE_FORMATS = {
    "DD/MM/YYYY": "%d/%m/%Y %H:%M",
    "MM/DD/YYYY": "%m/%d/%Y %H:%M",
    "YYYY-MM-DD": "%Y-%m-%d %H:%M",
}
SOURCE_DATE_FMT = "%d/%m/%Y %H:%M"


def format_for_display(df: pd.DataFrame, date_fmt: str, unit: str) -> pd.DataFrame:
    """Apply date reformatting and distance unit conversion for display."""
    df = df.copy()
    # Date formatting
    if "DateTimeOriginal" in df.columns and date_fmt != "DD/MM/YYYY":
        out_fmt = DATE_FORMATS[date_fmt]
        df["DateTimeOriginal"] = pd.to_datetime(
            df["DateTimeOriginal"], format=SOURCE_DATE_FMT, errors="coerce"
        ).dt.strftime(out_fmt)
    # Distance unit conversion
    if "distance_m" in df.columns and unit == "feet":
        df["distance_ft"] = df["distance_m"].apply(
            lambda d: round(d * 3.28084, 2) if pd.notna(d) else None
        )
        df.drop(columns=["distance_m"], inplace=True)
    return df


def compute_sample_point(left: int, top: int, right: int, bottom: int, mode: str) -> Tuple[int, int]:
    cx = int((left + right) / 2)
    cy = int((top + bottom) / 2)
    if mode == "bottom-center":
        return (cx, bottom)
    return (cx, cy)


def draw_overlays(
    img: Image.Image,
    df: pd.DataFrame,
    point_mode: str,
    show_boxes: bool = True,
    show_points: bool = True,
) -> Image.Image:
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    required = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
    if not all(c in df.columns for c in required):
        return annotated

    for _, row in df.iterrows():
        left = _safe_int(row.get("bbox_left"))
        top = _safe_int(row.get("bbox_top"))
        right = _safe_int(row.get("bbox_right"))
        bottom = _safe_int(row.get("bbox_bottom"))
        if None in (left, top, right, bottom):
            continue

        if show_boxes:
            draw.rectangle([(left, top), (right, bottom)], width=3)

            label = str(row.get("label", "")).strip()
            conf = row.get("confidence", None)
            if conf is not None and conf != "":
                try:
                    label = f"{label} ({float(conf):.2f})"
                except (TypeError, ValueError):
                    pass

            if label:
                draw.text((left, max(0, top - 14)), label, font=font)

        if show_points:
            x, y = compute_sample_point(left, top, right, bottom, point_mode)
            r = 6
            draw.ellipse([(x - r, y - r), (x + r, y + r)], width=3)

    return annotated


def draw_calibration_markers(img: Image.Image, calib_df: pd.DataFrame) -> Image.Image:
    """Draw calibration point markers on a copy of the image."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for _, row in calib_df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        dist = row.get("distance_m", None)
        r = 8
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill="red", outline="white", width=2)
        if dist is not None:
            label = f"{float(dist):.1f}m"
            draw.text((x + r + 4, y - 7), label, fill="yellow", font=font)

    return annotated


def scale_click_to_original(click: dict, original_size: Tuple[int, int]) -> Tuple[int, int]:
    """Convert displayed-image click coordinates to original image coordinates."""
    orig_w, orig_h = original_size
    disp_w = click.get("width", orig_w)
    disp_h = click.get("height", orig_h)

    x = int(click["x"] * orig_w / disp_w) if disp_w > 0 else int(click["x"])
    y = int(click["y"] * orig_h / disp_h) if disp_h > 0 else int(click["y"])

    x = max(0, min(x, orig_w - 1))
    y = max(0, min(y, orig_h - 1))
    return (x, y)


# ---------- depth estimation ----------
@st.cache_resource
def load_depth_model(model_id: str):
    """Load DepthAnything V2 model and processor. Cached across reruns."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    return processor, model


@st.cache_data
def compute_depth_map(image_path: str, model_id: str) -> np.ndarray:
    """Run DepthAnything V2 on an image. Returns (H, W) float32 depth array at original resolution."""
    img = Image.open(image_path).convert("RGB")
    processor, model = load_depth_model(model_id)

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, h, w)

    # Interpolate to original image size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(img.height, img.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().numpy()

    return depth


def colorize_depth(depth: np.ndarray) -> Image.Image:
    """Convert a 2D depth array to a colorized PIL Image using a viridis-like colormap."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 0:
        normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros(depth.shape, dtype=np.uint8)

    # Viridis-like lookup table built from anchor points
    lut = np.zeros((256, 3), dtype=np.uint8)
    anchors = [
        (0, 68, 1, 84),
        (64, 59, 82, 139),
        (128, 33, 145, 140),
        (192, 94, 201, 98),
        (255, 253, 231, 37),
    ]
    for i in range(len(anchors) - 1):
        i0, r0, g0, b0 = anchors[i]
        i1, r1, g1, b1 = anchors[i + 1]
        for j in range(i0, i1 + 1):
            t = (j - i0) / (i1 - i0)
            lut[j] = [int(r0 + t * (r1 - r0)), int(g0 + t * (g1 - g0)), int(b0 + t * (b1 - b0))]

    colored = lut[normalized]
    return Image.fromarray(colored)


# ---------- calibration ----------
def fit_calibration(calib_df: pd.DataFrame, depth_map: np.ndarray) -> Optional[dict]:
    """Fit distance = a / depth + b from calibration points. Returns None if < 2 valid points."""
    if len(calib_df) < 2:
        return None

    depth_values = []
    distances = []

    for _, row in calib_df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        d = float(row["distance_m"])

        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            dv = float(depth_map[y, x])
            if dv > 0:
                depth_values.append(dv)
                distances.append(d)

    if len(depth_values) < 2:
        return None

    depth_values = np.array(depth_values)
    distances = np.array(distances)

    # Fit: distance = a * (1/depth) + b
    inv_depth = 1.0 / depth_values
    A = np.column_stack([inv_depth, np.ones_like(inv_depth)])
    result, _, _, _ = np.linalg.lstsq(A, distances, rcond=None)
    a, b = result

    # R-squared
    predicted = a * inv_depth + b
    ss_res = np.sum((distances - predicted) ** 2)
    ss_tot = np.sum((distances - np.mean(distances)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "a": float(a),
        "b": float(b),
        "r_squared": float(r_squared),
        "n_points": len(depth_values),
    }


def apply_calibration(depth_value: float, calib: dict) -> float:
    """Convert a single depth value to distance in meters using fitted calibration."""
    if depth_value > 0:
        return calib["a"] / depth_value + calib["b"]
    return float("nan")


# ---------- calibration file I/O ----------
def load_calibration_points() -> pd.DataFrame:
    """Load calibration points from CSV if it exists; otherwise return empty DF."""
    cols = ["x", "y", "distance_m"]
    if os.path.exists(CALIBRATION_CSV_PATH):
        try:
            df = pd.read_csv(CALIBRATION_CSV_PATH)
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            return df[cols].copy()
        except (pd.errors.ParserError, KeyError, ValueError):
            return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)


def save_calibration_points(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(CALIBRATION_CSV_PATH), exist_ok=True)
    df.to_csv(CALIBRATION_CSV_PATH, index=False)


# ---------- load data ----------
if not os.path.exists(CSV_PATH):
    st.error(f"Detections CSV not found: {CSV_PATH}")
    st.stop()

predictions = pd.read_csv(CSV_PATH)

images = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
if not images:
    st.error(f"No images found in {IMAGE_FOLDER}")
    st.stop()

if "relative_path" not in predictions.columns:
    st.error("CSV missing required column: relative_path")
    st.stop()

_defaults = {
    "calib_df": load_calibration_points(),
    "last_click": None,
    "last_click_time": 0,
    "depth_map": None,
    "depth_image": None,
    "calibration": None,
    "results_with_distances": None,
    "point_mode": "bottom-center",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------- UI layout ----------
controls_col, preview_col = st.columns([0.30, 0.70], gap="large")

with controls_col:
    st.subheader("Step 1: Select an Image")
    st.caption("Choose a camera-trap image from your deployment to work with.")

    selected_image = st.selectbox(
        "Image",
        images,
        help="Pick an image from your deployment folder to analyse.",
    )

    # --- Depth estimation section ---
    st.divider()
    st.subheader("Step 2: Generate Depth Map")
    st.caption("Estimate per-pixel depth and produce a colour-coded depth map.")

    model_choice = st.selectbox(
        "Model size",
        list(DEPTH_MODELS.keys()),
        index=0,
        help="Larger models are slower but produce more accurate depth maps. 'Small' is fine for most images.",
    )
    model_id = DEPTH_MODELS[model_choice]

    if st.button("Generate Depth Map", use_container_width=True, type="primary"):
        img_path = os.path.join(IMAGE_FOLDER, selected_image)
        with st.spinner("Loading model and computing depth map..."):
            depth_map = compute_depth_map(img_path, model_id)
            st.session_state.depth_map = depth_map
            st.session_state.depth_image = selected_image
            # Clear stale calibration fit when regenerating
            st.session_state.calibration = None
            st.session_state.results_with_distances = None

    if st.session_state.depth_map is not None:
        st.success(f"Depth map ready for {st.session_state.depth_image}.")
    else:
        st.info("Press the button above to generate a depth map.")

    # --- Overlay toggles ---
    st.divider()
    st.caption("**Display options** -- control what is shown on the preview image.")
    show_boxes = st.toggle(
        "Show bounding boxes",
        value=True,
        help="Show rectangles around detected animals in the preview.",
    )
    show_points = st.toggle(
        "Show sampling points",
        value=True,
        help="Show the specific pixel used to measure each animal's depth.",
    )

    _modes = ["bottom-center", "center"]
    point_mode = st.radio(
        "Sampling point position",
        _modes,
        horizontal=True,
        index=_modes.index(st.session_state.point_mode),
        help="Where on each animal's bounding box to measure depth. "
             "'bottom-center' samples at the animal's feet (usually best for ground distance). "
             "'center' samples at the middle of the bounding box.",
    )
    st.session_state.point_mode = point_mode

    # --- Display preferences ---
    st.divider()
    st.caption("**Output preferences** -- affects the detections table and exported CSV.")
    distance_unit = st.radio(
        "Distance unit",
        ["meters", "feet"],
        horizontal=True,
        key="distance_unit",
        help="Unit for computed distances in the table and downloaded CSV.",
    )
    date_format = st.selectbox(
        "Date format",
        ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"],
        key="date_format",
        help="How dates from image metadata are displayed in the table.",
    )

    # Filter detections for selected image
    df_img = predictions[predictions["relative_path"] == selected_image].copy()

    if "confidence" in df_img.columns:
        df_img["confidence"] = pd.to_numeric(df_img["confidence"], errors="coerce")
        df_img = df_img.sort_values("confidence", ascending=False)

    label_filter = "All"
    if "label" in df_img.columns and len(df_img) > 0:
        labels = sorted(df_img["label"].astype(str).unique().tolist())
        label_filter = st.selectbox(
            "Filter by label",
            ["All"] + labels,
            help="Show only detections matching a specific species or category.",
        )

    if label_filter != "All":
        df_show = df_img[df_img["label"].astype(str) == label_filter].copy()
    else:
        df_show = df_img.copy()

    # Add sample point columns
    required_bbox_cols = ["bbox_left", "bbox_top", "bbox_right", "bbox_bottom"]
    if len(df_show) > 0 and all(c in df_show.columns for c in required_bbox_cols):
        xs, ys = [], []
        for _, row in df_show.iterrows():
            left_i = _safe_int(row.get("bbox_left"))
            top_i = _safe_int(row.get("bbox_top"))
            right_i = _safe_int(row.get("bbox_right"))
            bottom_i = _safe_int(row.get("bbox_bottom"))
            if None in (left_i, top_i, right_i, bottom_i):
                xs.append(None)
                ys.append(None)
                continue
            x, y = compute_sample_point(left_i, top_i, right_i, bottom_i, point_mode)
            xs.append(x)
            ys.append(y)
        df_show["sample_x"] = xs
        df_show["sample_y"] = ys
        dm = st.session_state.get("depth_map")
        if dm is not None and st.session_state.get("depth_image") == selected_image:
            dvs = []
            for sx, sy in zip(xs, ys):
                if sx is not None and sy is not None:
                    sy_c = max(0, min(sy, dm.shape[0] - 1))
                    sx_c = max(0, min(sx, dm.shape[1] - 1))
                    dvs.append(round(float(dm[sy_c, sx_c]), 4))
                else:
                    dvs.append(None)
            df_show["depth_value"] = dvs

    st.caption(f"{len(df_show)} detections shown (of {len(df_img)} total)")

    if st.session_state.depth_map is None:
        st.info("**Next:** Generate a depth map above (Step 2) to proceed with calibration.")
    elif st.session_state.calibration is None and len(st.session_state.calib_df) < 2:
        st.info("**Next:** Add at least 2 calibration points below (Step 3) to enable distance estimation.")

    # --- Calibration section ---
    st.divider()
    st.subheader("Step 3: Calibrate Distances")
    st.caption(
        "The depth map gives relative depth (closer vs. farther) but not real-world distances. "
        "To convert depth into meters, tell the tool the actual distance to a few reference points "
        "in the image -- e.g. a tree 5 m away or a post at 15 m. "
        "Add at least 2 points at different distances for a good calibration."
    )
    st.info(
        "Click on the preview image to mark a point, enter its known distance below, "
        "then press **Add calibration point**."
    )

    dist_in = st.number_input(
        "Known distance (m)",
        min_value=0.0,
        value=5.0,
        step=0.5,
        help="The real-world distance in meters from the camera to the point you clicked on the image.",
    )

    last = st.session_state.last_click
    if last is None:
        st.info("Click on the preview image to select a calibration point.")
    else:
        depth_at_point = ""
        if st.session_state.depth_map is not None:
            dm = st.session_state.depth_map
            if 0 <= last[1] < dm.shape[0] and 0 <= last[0] < dm.shape[1]:
                depth_at_point = f" | depth value: {dm[last[1], last[0]]:.4f}"
        st.success(f"Selected point: x={last[0]}, y={last[1]}{depth_at_point}")

    add_clicked = st.button("Add calibration point", use_container_width=True)

    if add_clicked:
        d_val = _safe_float(dist_in)
        if last is None:
            st.error("Please click on the preview image first.")
        elif d_val is None or d_val <= 0:
            st.error("Please enter a distance > 0 meters.")
        else:
            new_row = pd.DataFrame([{"x": int(last[0]), "y": int(last[1]), "distance_m": float(d_val)}])
            st.session_state.calib_df = pd.concat([st.session_state.calib_df, new_row], ignore_index=True)
            st.session_state.results_with_distances = None
            st.success(f"Added point: ({int(last[0])}, {int(last[1])}) → {float(d_val)} m")

    st.caption("Your calibration points so far:")
    st.dataframe(st.session_state.calib_df, width="stretch")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save", use_container_width=True):
            save_calibration_points(st.session_state.calib_df)
            st.success(f"Saved to {CALIBRATION_CSV_PATH}")
    with b2:
        if st.button("Reload from disk", use_container_width=True):
            st.session_state.calib_df = load_calibration_points()
            st.session_state.results_with_distances = None
            st.success("Reloaded calibration points from disk")

    if st.button(
        "Clear all calibration points",
        use_container_width=True,
        help="Removes all calibration points from memory. Does not affect the saved file until you press Save.",
    ):
        st.session_state.calib_df = pd.DataFrame(columns=["x", "y", "distance_m"])
        st.session_state.last_click = None
        st.session_state.calibration = None
        st.session_state.results_with_distances = None
        st.warning("Cleared calibration points (not saved until you click Save).")

    # --- Calibration fit ---
    if st.session_state.depth_map is not None and len(st.session_state.calib_df) >= 2:
        st.divider()
        st.subheader("Calibration Quality")
        st.caption(
            "Shows how well the calibration fits your reference points. "
            "A higher accuracy score (closer to 1.0) means more reliable distance estimates."
        )

        calib_result = fit_calibration(st.session_state.calib_df, st.session_state.depth_map)
        if calib_result is not None:
            st.session_state.calibration = calib_result
            st.metric(
                "Accuracy score (R\u00b2)",
                f"{calib_result['r_squared']:.4f}",
                help="R-squared measures how well the calibration fits your reference points. "
                     "1.0 is a perfect fit. Values above 0.9 are good; below 0.8 suggests you need "
                     "more points or more varied distances.",
            )
            st.caption(f"Calibration formula: distance = {calib_result['a']:.4f} / depth + {calib_result['b']:.4f}")
            st.caption(f"Fitted on {calib_result['n_points']} points")

            if calib_result["r_squared"] < 0.8:
                st.warning(
                    "Low accuracy score. The calibration may produce unreliable distances. "
                    "Try adding more reference points at a wider range of distances (e.g. one close and one far away)."
                )
        else:
            st.error("Could not fit calibration. Check that points have valid depth values > 0.")

    # --- Add distance column to df_show (after calibration is fitted) ---
    calib = st.session_state.get("calibration")
    if calib is not None and "depth_value" in df_show.columns:
        dists = []
        for d in df_show["depth_value"]:
            if d is None:
                dists.append(None)
            else:
                val = apply_calibration(d, calib)
                dists.append(round(val, 2) if not np.isnan(val) else None)
        df_show["distance_m"] = dists

    # --- Compute distances ---
    calibration = st.session_state.get("calibration")
    depth_map_state = st.session_state.get("depth_map")

    if calibration is not None and depth_map_state is not None:
        st.divider()
        st.subheader("Step 4: Compute Distances")
        st.caption(
            "Apply your calibration to estimate the distance from the camera to every "
            "detected animal across all images. A depth map is computed per image "
            "(cached after the first run)."
        )

        if st.button(
            "Compute distances for all detections",
            use_container_width=True,
            type="primary",
            help="Uses a per-image depth map and your calibration to estimate how far each detected animal is from the camera.",
        ):
            results = predictions.copy()
            sample_xs, sample_ys, depth_values, distances = [], [], [], []

            # Pre-compute depth maps keyed by relative_path
            unique_images = results["relative_path"].unique()
            depth_maps: dict[str, np.ndarray] = {}
            progress = st.progress(0, text="Computing depth maps...")
            skipped_images: list[str] = []

            for idx, rel_path in enumerate(unique_images):
                img_full = os.path.join(IMAGE_FOLDER, rel_path)
                if os.path.isfile(img_full):
                    depth_maps[rel_path] = compute_depth_map(img_full, model_id)
                else:
                    skipped_images.append(rel_path)
                progress.progress(
                    (idx + 1) / len(unique_images),
                    text=f"Depth maps: {idx + 1}/{len(unique_images)}",
                )

            progress.empty()

            if skipped_images:
                st.warning(
                    f"Skipped {len(skipped_images)} image(s) not found on disk: "
                    + ", ".join(skipped_images[:5])
                    + ("..." if len(skipped_images) > 5 else "")
                )

            for _, row in results.iterrows():
                rel_path = row.get("relative_path")
                dm = depth_maps.get(rel_path)

                if dm is None:
                    sample_xs.append(None)
                    sample_ys.append(None)
                    depth_values.append(None)
                    distances.append(None)
                    continue

                left = _safe_int(row.get("bbox_left"))
                top = _safe_int(row.get("bbox_top"))
                right = _safe_int(row.get("bbox_right"))
                bottom = _safe_int(row.get("bbox_bottom"))

                if None in (left, top, right, bottom):
                    sample_xs.append(None)
                    sample_ys.append(None)
                    depth_values.append(None)
                    distances.append(None)
                    continue

                x, y = compute_sample_point(left, top, right, bottom, point_mode)
                y = max(0, min(y, dm.shape[0] - 1))
                x = max(0, min(x, dm.shape[1] - 1))

                dv = float(dm[y, x])
                dist = apply_calibration(dv, calibration)
                sample_xs.append(x)
                sample_ys.append(y)
                depth_values.append(round(dv, 4))
                distances.append(round(dist, 2) if not np.isnan(dist) else None)

            results["sample_x"] = sample_xs
            results["sample_y"] = sample_ys
            results["depth_value"] = depth_values
            results["distance_m"] = distances
            st.session_state.results_with_distances = results
            n_valid = sum(d is not None for d in distances)
            st.success(f"Computed distances for {n_valid} / {len(distances)} detections.")

        if st.session_state.results_with_distances is not None:
            csv_data = format_for_display(st.session_state.results_with_distances, date_format, distance_unit).to_csv(index=False)
            st.download_button(
                "Download results CSV",
                csv_data,
                file_name="results_with_distances.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------- preview column ----------
with preview_col:
    st.subheader("Image Preview")
    st.markdown("---")

    img_path = os.path.join(IMAGE_FOLDER, selected_image)
    img = Image.open(img_path).convert("RGB")
    original_size = img.size  # (width, height)

    # Build annotated RGB preview
    preview_rgb = img.copy()
    if len(df_show) > 0 and (show_boxes or show_points):
        preview_rgb = draw_overlays(img, df_show, point_mode, show_boxes, show_points)
    if len(st.session_state.calib_df) > 0:
        preview_rgb = draw_calibration_markers(preview_rgb, st.session_state.calib_df)

    depth_map_active = (st.session_state.depth_map is not None
                        and st.session_state.depth_image == selected_image)

    if depth_map_active:
        depth_colored = colorize_depth(st.session_state.depth_map)
        if len(df_show) > 0 and (show_boxes or show_points):
            depth_colored = draw_overlays(depth_colored, df_show, point_mode, show_boxes, show_points)
        if len(st.session_state.calib_df) > 0:
            depth_colored = draw_calibration_markers(depth_colored, st.session_state.calib_df)

        # Draggable comparison slider on the image
        image_comparison(
            img1=preview_rgb,
            img2=depth_colored,
            label1="RGB",
            label2="Depth",
            width=original_size[0],
            starting_position=50,
            make_responsive=True,
            in_memory=True,
        )
        st.caption("Drag the slider left/right to compare the original photo with the depth map.")

        # Calibration clicks via a separate clickable image
        with st.expander("Click here to add a calibration point (opens a clickable copy of the image)"):
            click = streamlit_image_coordinates(
                preview_rgb,
                key="composite_click",
                cursor="crosshair",
                image_format="JPEG",
                jpeg_quality=85,
            )
            if click is not None:
                click_time = click.get("unix_time", 0)
                if click_time > st.session_state.last_click_time:
                    st.session_state.last_click_time = click_time
                    st.session_state.last_click = scale_click_to_original(click, original_size)

    else:
        # Single image view (no depth map yet)
        st.caption("Click anywhere on the image to mark a calibration point. Then enter its distance in the sidebar.")
        click = streamlit_image_coordinates(
            preview_rgb,
            key="preview_click",
            cursor="crosshair",
            image_format="JPEG",
            jpeg_quality=85,
        )
        if click is not None:
            click_time = click.get("unix_time", 0)
            if click_time > st.session_state.last_click_time:
                st.session_state.last_click_time = click_time
                st.session_state.last_click = scale_click_to_original(click, original_size)

    st.caption(f"Viewing: {selected_image} | Sampling point: {st.session_state.point_mode}")

# ---------- detections table ----------
st.subheader("Detections Table")
st.caption("Showing up to 10 sample detections. Distance columns appear after you complete calibration and compute distances.")
if st.session_state.results_with_distances is not None:
    results_df = st.session_state.results_with_distances
    st.dataframe(format_for_display(results_df.head(10), date_format, distance_unit), width="stretch")
else:
    st.dataframe(format_for_display(df_show.head(10), date_format, distance_unit), width="stretch")
