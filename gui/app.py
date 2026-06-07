"""Flask-based GUI for AestheticCropper."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import time
import json
from pathlib import Path

# Fix ultralytics settings permission issue: set YOLO_SETTINGS_DIR before any imports
_yolo_settings_dir = str(Path.home() / ".config" / "ultralytics")
os.makedirs(_yolo_settings_dir, exist_ok=True)
os.environ.setdefault("YOLO_SETTINGS_DIR", _yolo_settings_dir)

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import AestheticCropper
from src.utils import load_config, draw_bbox, load_image


class NumpyJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that handles numpy types."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

# Global cropper instance (initialized on first request)
cropper: AestheticCropper = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_cropper() -> AestheticCropper:
    """Lazy-initialize the AestheticCropper instance."""
    global cropper
    if cropper is None:
        config_path = os.environ.get("AESTHETIC_CROPPER_CONFIG")
        if not config_path:
            # Resolve relative to the smart-framing project root
            project_root = Path(__file__).resolve().parent.parent
            config_path = str(project_root / "config.yaml")
        cropper = AestheticCropper(config_path=config_path)
    return cropper


def image_to_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode an OpenCV BGR image to a base64 data URL."""
    _, buf = cv2.imencode(fmt, image)
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/crop", methods=["POST"])
def crop_image():
    """Process a single uploaded image and return results."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read uploaded file into temp file (auto-deleted after processing)
    ext = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Process
        cr = get_cropper()
        result = cr.process(tmp_path)

        # Prepare visualization
        image = load_image(tmp_path)
        vis = draw_bbox(image, result.best_bbox, f"score={result.best_score:.3f}")

        # Prepare response
        response = {
            "bbox": [int(x) for x in result.best_bbox],
            "score": float(round(float(result.best_score), 4)),
            "explanation": result.explanation,
            "sub_scores": {
                "aesthetic": float(round(float(result.best_sub_scores.aesthetic), 4)),
                "saliency": float(round(float(result.best_sub_scores.saliency), 4)),
                "composition": float(round(float(result.best_sub_scores.composition), 4)),
                "subject": float(round(float(result.best_sub_scores.subject), 4)),
                "technical": float(round(float(result.best_sub_scores.technical), 4)),
                "area_prior": float(round(float(result.best_sub_scores.area_prior), 4)),
                "thirds": float(round(float(result.best_sub_scores.thirds), 4)),
                "center_balance": float(round(float(result.best_sub_scores.center_balance), 4)),
                "whitespace": float(round(float(result.best_sub_scores.whitespace), 4)),
                "edge_simplicity": float(round(float(result.best_sub_scores.edge_simplicity), 4)),
                "symmetry": float(round(float(result.best_sub_scores.symmetry), 4)),
                "sharpness": float(round(float(result.best_sub_scores.sharpness), 4)),
                "brightness": float(round(float(result.best_sub_scores.brightness), 4)),
                "contrast": float(round(float(result.best_sub_scores.contrast), 4)),
                "saturation": float(round(float(result.best_sub_scores.saturation), 4)),
            },
            "original_image": image_to_base64(vis),
            "crop_image": image_to_base64(result.best_crop),
            "top_candidates": [
                {
                    "bbox": [int(x) for x in c.bbox],
                    "score": float(round(float(c.final_score), 4)),
                }
                for c in result.top_candidates
            ],
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/batch", methods=["POST"])
def batch_process():
    """Process multiple uploaded images in batch."""
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    cr = get_cropper()

    for file in files:
        if file.filename == "":
            continue

        ext = Path(file.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = cr.process(tmp_path)
            image = load_image(tmp_path)
            vis = draw_bbox(image, result.best_bbox, f"score={result.best_score:.3f}")
            results.append({
                "filename": file.filename,
                "bbox": [int(x) for x in result.best_bbox],
                "score": float(round(float(result.best_score), 4)),
                "explanation": result.explanation,
                "original_image": image_to_base64(vis),
                "crop_image": image_to_base64(result.best_crop),
                "sub_scores": {
                    "aesthetic": float(round(float(result.best_sub_scores.aesthetic), 4)),
                    "saliency": float(round(float(result.best_sub_scores.saliency), 4)),
                    "composition": float(round(float(result.best_sub_scores.composition), 4)),
                    "subject": float(round(float(result.best_sub_scores.subject), 4)),
                    "technical": float(round(float(result.best_sub_scores.technical), 4)),
                    "area_prior": float(round(float(result.best_sub_scores.area_prior), 4)),
                },
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return jsonify({"results": results})


@app.route("/api/export", methods=["POST"])
def export_coordinates():
    """Export cropping coordinates as JSON/CSV."""
    data = request.get_json()
    if not data or "results" not in data:
        return jsonify({"error": "No results data provided"}), 400

    results = data["results"]
    # Return as downloadable JSON
    return jsonify({"coordinates": results})


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return current configuration."""
    try:
        config = load_config()
        return jsonify(config)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update configuration (weights, etc.)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No config data provided"}), 400

    global cropper
    # Reset cropper to pick up changes
    cropper = None

    return jsonify({"status": "ok"})


def main():
    """Run the Flask server."""
    import argparse

    parser = argparse.ArgumentParser(description="AestheticCropper GUI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", type=str, default="", help="Config file path")
    args = parser.parse_args()

    if args.config:
        os.environ["AESTHETIC_CROPPER_CONFIG"] = args.config

    print(f"Starting AestheticCropper GUI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
