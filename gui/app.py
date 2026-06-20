"""Flask-based GUI for AestheticCropper."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import time
import json
import csv
from pathlib import Path
from flask import Response
import zipfile

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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.gaic.yaml"


def get_cropper() -> AestheticCropper:
    """Lazy-initialize the AestheticCropper instance."""
    global cropper
    if cropper is None:
        config_path = os.environ.get("AESTHETIC_CROPPER_CONFIG")
        if not config_path:
            config_path = str(DEFAULT_CONFIG_PATH)
        cropper = AestheticCropper(config_path=config_path)
    return cropper

current_weights = None

@app.route('/api/config', methods=['POST'])
def update_config():
    global cropper, current_weights
    data = request.get_json()
    if 'weights' in data:
        current_weights = data['weights']
        if cropper:
            # 更新 fusion 模块的权重
            for k, v in current_weights.items():
                setattr(cropper.fusion, f'weight_{k}', v)
            # 重新归一化
            total = sum(current_weights.values())
            if total > 0:
                for k in current_weights:
                    setattr(cropper.fusion, f'weight_{k}', current_weights[k] / total)
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'Invalid config'}), 400


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
                    "crop_base64": image_to_base64(image[c.bbox[1]:c.bbox[3], c.bbox[0]:c.bbox[2]])
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


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return current configuration."""
    try:
        config_path = os.environ.get("AESTHETIC_CROPPER_CONFIG", str(DEFAULT_CONFIG_PATH))
        config = load_config(config_path)
        return jsonify(config)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def export_coordinates():
    data = request.get_json()
    results = data.get("results", [])
    fmt = data.get("format", "json")  # 'json' or 'csv'
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output, fieldnames=["image", "x1", "y1", "x2", "y2", "score"]
        )
        writer.writeheader()
        for r in results:
            # 注意：根据实际数据字段名，可能是 'filename' 或 'image'
            writer.writerow(
                {
                    "image": r.get("filename", r.get("image", "")),
                    "x1": r["bbox"][0],
                    "y1": r["bbox"][1],
                    "x2": r["bbox"][2],
                    "y2": r["bbox"][3],
                    "score": r.get("score", 0),
                }
            )
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=predictions.csv"},
        )
    else:
        # 返回 JSON 格式
        return jsonify(results)


@app.route("/api/batch_download", methods=["POST"])
def batch_download():
    data = request.get_json()
    results = data.get("results", [])
    if not results:
        return jsonify({"error": "No results provided"}), 400
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in results:
            crop_base64 = item.get("crop_image")
            filename = item.get("filename", "crop")
            if crop_base64 and crop_base64.startswith("data:image"):
                # 提取 base64 数据
                img_data = base64.b64decode(crop_base64.split(",")[1])
                zip_file.writestr(f"{filename}_crop.jpg", img_data)
    zip_buffer.seek(0)
    return Response(
        zip_buffer.read(),
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment;filename=crops.zip"},
    )


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
