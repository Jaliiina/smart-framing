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

_yolo_settings_dir = str(Path.home() / ".config" / "ultralytics")
os.makedirs(_yolo_settings_dir, exist_ok=True)
os.environ.setdefault("YOLO_SETTINGS_DIR", _yolo_settings_dir)

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask import stream_with_context
import urllib.parse

try:
    from flask.json.provider import DefaultJSONProvider

    _HAS_JSON_PROVIDER = True
except ImportError:
    DefaultJSONProvider = object
    _HAS_JSON_PROVIDER = False
    from flask.json import JSONEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import AestheticCropper
from src.utils import load_config, draw_bbox, load_image


if _HAS_JSON_PROVIDER:
    class NumpyJSONProvider(DefaultJSONProvider):

        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
else:
    class NumpyJSONEncoder(JSONEncoder):

        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)


app = Flask(__name__)
if _HAS_JSON_PROVIDER:
    app.json_provider_class = NumpyJSONProvider
    app.json = NumpyJSONProvider(app)
else:
    app.json_encoder = NumpyJSONEncoder
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

cropper: AestheticCropper = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.gaic.yaml"


def get_config_path() -> str:
    return os.environ.get("AESTHETIC_CROPPER_CONFIG", str(DEFAULT_CONFIG_PATH))


def get_cropper() -> AestheticCropper:
    """Lazy-initialize the AestheticCropper instance."""
    global cropper
    if cropper is None:
        cropper = AestheticCropper(config_path=get_config_path())
    return cropper


def convert_to_clip_prompts(text: str):
    if not text:
        return []

    try:
        from src.llm_crop_explainer import LLMCropExplainer
        cfg = load_config(get_config_path()).get('llm', {})
        try:
            llm = LLMCropExplainer(cfg)
        except Exception:
            llm = None

        if llm and getattr(llm, 'enable', False) and getattr(llm, 'api_key', None):
            instr = (
                "You are an expert at writing CLIP-style prompts.\n"
                "Convert the following user input (which may be Chinese) into a list of 5-8 concise, English, CLIP-friendly prompt phrases.\n"
                "Each prompt should be short (3-7 words), focused, and suitable for CLIP text encoding (no full sentences).\n"
                "Return ONLY a JSON array of strings, e.g. [\"prompt1\", \"prompt2\"], with no extra explanation.\n"
                f"User input: {text}"
            )
            try:
                resp = llm.chat(instr)
                try:
                    parsed = json.loads(resp)
                    if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                        return parsed
                except Exception:
                    import re

                    m = re.search(r"\[.*\]", resp, re.S)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                            if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                                return parsed
                        except Exception:
                            pass
                    parts = [p.strip().strip('"') for p in re.split(r"\n|\||;|,", resp) if p.strip()]
                    parts = [p for p in parts if 2 <= len(p.split()) <= 8]
                    if parts:
                        return parts[:8]
            except Exception:
                app.logger.exception('LLM conversion to CLIP prompts failed')

    except Exception:
        pass

    base = text.strip()
    variants = [
        f"a high quality photograph of {base}",
        f"a close-up shot of {base}",
        f"a well composed photo of {base}",
        f"a clean composition featuring {base}",
        f"a detailed image of {base} with good lighting",
    ]
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


current_weights = None


@app.route("/api/config", methods=["POST"])
def update_config():
    global cropper, current_weights
    data = request.get_json()
    if "weights" in data:
        current_weights = data["weights"]
        if cropper:
            for k, v in current_weights.items():
                setattr(cropper.fusion, f"weight_{k}", v)
            total = sum(current_weights.values())
            if total > 0:
                for k in current_weights:
                    setattr(cropper.fusion, f"weight_{k}", current_weights[k] / total)
        return jsonify({"status": "ok"})
    return jsonify({"error": "Invalid config"}), 400


def image_to_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    _, buf = cv2.imencode(fmt, image)
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/crop", methods=["POST"])
def crop_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    ext = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    user_prompt = request.form.get('user_prompt') if request.form else None
    clip_prompts_raw = request.form.get('clip_prompts') if request.form else None
    clip_prompts = None
    if clip_prompts_raw:
        try:
            clip_prompts = json.loads(clip_prompts_raw)
        except Exception:
            clip_prompts = None

    try:
        cr = get_cropper()
        original_top_k = getattr(cr.fusion, 'top_k_display', None)
        try:
            cr.fusion.top_k_display = 5
        except Exception:
            original_top_k = None
        original_semantic_prompts = None
        try:
            prompts_to_use = None
            if clip_prompts is not None:
                prompts_to_use = clip_prompts
            elif user_prompt:
                prompts_to_use = convert_to_clip_prompts(user_prompt)

            if prompts_to_use:
                scs = getattr(cr, 'semantic_crop_scorer', None)
                if scs and hasattr(scs, 'positive_prompts'):
                    original_semantic_prompts = list(scs.positive_prompts)
                    merged_prompts = original_semantic_prompts + list(prompts_to_use)
                    if hasattr(scs, 'set_positive_prompts'):
                        scs.set_positive_prompts(merged_prompts)
                    else:
                        scs.positive_prompts = merged_prompts
        except Exception:
            app.logger.exception('Failed to append user CLIP prompts, continuing without them')

        result = cr.process(tmp_path)

        image = load_image(tmp_path)
        h, w = image.shape[:2]


        saliency_map = result.saliency_map
        if saliency_map is not None:
            saliency_uint8 = (saliency_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
            alpha = 0.5
            overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
            saliency_vis = image_to_base64(overlay)
        else:
            saliency_vis = None

        vis_obj = image.copy()
        for obj in result.detected_objects:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(vis_obj, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{obj.class_name} {obj.confidence:.2f}"
            cv2.putText(
                vis_obj,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        object_vis = image_to_base64(vis_obj)

        vis = draw_bbox(image, result.best_bbox, f"score={result.best_score:.3f}")

        response = {
            "bbox": [int(x) for x in result.best_bbox],
            "score": float(round(float(result.best_score), 4)),
            "explanation": result.explanation,
            "explanation_full": getattr(result, 'explanation_full', '') or '',
            "sub_scores": {
                "aesthetic": float(round(float(result.best_sub_scores.aesthetic), 4)),
                "saliency": float(round(float(result.best_sub_scores.saliency), 4)),
                "composition": float(
                    round(float(result.best_sub_scores.composition), 4)
                ),
                "subject": float(round(float(result.best_sub_scores.subject), 4)),
                "technical": float(round(float(result.best_sub_scores.technical), 4)),
                "area_prior": float(round(float(result.best_sub_scores.area_prior), 4)),
                "thirds": float(round(float(result.best_sub_scores.thirds), 4)),
                "center_balance": float(
                    round(float(result.best_sub_scores.center_balance), 4)
                ),
                "whitespace": float(round(float(result.best_sub_scores.whitespace), 4)),
                "edge_simplicity": float(
                    round(float(result.best_sub_scores.edge_simplicity), 4)
                ),
                "symmetry": float(round(float(result.best_sub_scores.symmetry), 4)),
                "sharpness": float(round(float(result.best_sub_scores.sharpness), 4)),
                "brightness": float(round(float(result.best_sub_scores.brightness), 4)),
                "contrast": float(round(float(result.best_sub_scores.contrast), 4)),
                "saturation": float(round(float(result.best_sub_scores.saturation), 4)),
            },
            "original_image": image_to_base64(vis),  
            "raw_image": image_to_base64(image),  
            "crop_image": image_to_base64(result.best_crop),
            "top_candidates": [
                {
                    "bbox": [int(x) for x in c.bbox],
                    "score": float(round(float(c.final_score), 4)),
                    "crop_base64": image_to_base64(
                        image[c.bbox[1] : c.bbox[3], c.bbox[0] : c.bbox[2]]
                    ),
                }
                for c in result.top_candidates
            ],
            "saliency_vis": saliency_vis,
            "object_vis": object_vis,
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
        try:
            if 'original_semantic_prompts' in locals() and original_semantic_prompts is not None:
                scs = getattr(cr, 'semantic_crop_scorer', None)
                if scs and hasattr(scs, 'positive_prompts'):
                    if hasattr(scs, 'set_positive_prompts'):
                        scs.set_positive_prompts(original_semantic_prompts)
                    else:
                        scs.positive_prompts = original_semantic_prompts
        except Exception:
            pass
        try:
            if original_top_k is not None:
                cr.fusion.top_k_display = original_top_k
        except Exception:
            pass


@app.route("/api/batch", methods=["POST"])
def batch_process():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    cr = get_cropper()
    original_top_k = getattr(cr.fusion, 'top_k_display', None)
    cr.fusion.top_k_display = 5  
    user_prompt = request.form.get('user_prompt') if request.form else None
    clip_prompts_raw = request.form.get('clip_prompts') if request.form else None
    clip_prompts = None
    if clip_prompts_raw:
        try:
            clip_prompts = json.loads(clip_prompts_raw)
        except Exception:
            clip_prompts = None

    original_semantic_prompts = None
    try:
        prompts_to_use = None
        if clip_prompts is not None:
            prompts_to_use = clip_prompts
        elif user_prompt:
            try:
                prompts_to_use = convert_to_clip_prompts(user_prompt)
            except Exception:
                prompts_to_use = None

        if prompts_to_use:
            scs = getattr(cr, 'semantic_crop_scorer', None)
            if scs and hasattr(scs, 'positive_prompts'):
                original_semantic_prompts = list(scs.positive_prompts)
                merged_prompts = original_semantic_prompts + list(prompts_to_use)
                if hasattr(scs, 'set_positive_prompts'):
                    scs.set_positive_prompts(merged_prompts)
                else:
                    scs.positive_prompts = merged_prompts

        for file in files:
            if not file or file.filename == "":
                continue

            ext = Path(file.filename).suffix or ".jpg"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    file.save(tmp.name)
                    tmp_path = tmp.name

                result = cr.process(tmp_path)
                image = load_image(tmp_path)
                vis = draw_bbox(image, result.best_bbox, f"score={result.best_score:.3f}")

                top_candidates = []
                for c in getattr(result, 'top_candidates', []):
                    try:
                        crop_img = image[c.bbox[1] : c.bbox[3], c.bbox[0] : c.bbox[2]]
                        vis_cand = draw_bbox(image.copy(), c.bbox, f"score={c.final_score:.3f}")
                        top_candidates.append({
                            "bbox": [int(x) for x in c.bbox],
                            "score": float(round(float(c.final_score), 4)),
                            "crop_base64": image_to_base64(crop_img),
                            "original_with_bbox": image_to_base64(vis_cand),
                        })
                    except Exception:
                        continue

                results.append({
                    "filename": file.filename,
                    "bbox": [int(x) for x in result.best_bbox],
                    "score": float(round(float(result.best_score), 4)),
                    "explanation": getattr(result, 'explanation', ''),
                    "explanation_full": getattr(result, 'explanation_full', '') or '',
                    "original_image": image_to_base64(vis),
                    "crop_image": image_to_base64(result.best_crop),
                    "top_candidates": top_candidates,
                    "sub_scores": {
                        "aesthetic": float(round(float(getattr(result.best_sub_scores, 'aesthetic', 0)), 4)),
                        "saliency": float(round(float(getattr(result.best_sub_scores, 'saliency', 0)), 4)),
                        "composition": float(round(float(getattr(result.best_sub_scores, 'composition', 0)), 4)),
                        "subject": float(round(float(getattr(result.best_sub_scores, 'subject', 0)), 4)),
                        "technical": float(round(float(getattr(result.best_sub_scores, 'technical', 0)), 4)),
                        "area_prior": float(round(float(getattr(result.best_sub_scores, 'area_prior', 0)), 4)),
                    },
                })
            except Exception as e:
                results.append({"filename": getattr(file, 'filename', 'unknown'), "error": str(e)})
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
    finally:
        try:
            if original_semantic_prompts is not None:
                scs = getattr(cr, 'semantic_crop_scorer', None)
                if scs and hasattr(scs, 'positive_prompts'):
                    if hasattr(scs, 'set_positive_prompts'):
                        scs.set_positive_prompts(original_semantic_prompts)
                    else:
                        scs.positive_prompts = original_semantic_prompts
        except Exception:
            app.logger.exception('Failed to restore semantic prompts after batch')

        if original_top_k is not None:
            try:
                cr.fusion.top_k_display = original_top_k
            except Exception:
                pass

    return jsonify({"results": results})


@app.route('/api/export_batch_report', methods=['POST'])
def export_batch_report():
    data = request.get_json(silent=True)
    if not data or 'results' not in data:
        return jsonify({'error': 'No results provided'}), 400
    results = data['results']

    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            combined = []
            for idx, r in enumerate(results):
                name_prefix = r.get('filename') or f'image_{idx+1}'
                orig_b64 = r.get('original_image')
                if orig_b64 and orig_b64.startswith('data:image'):
                    orig_data = base64.b64decode(orig_b64.split(',', 1)[1])
                    zf.writestr(f'{name_prefix}_original.jpg', orig_data)

                crop_b64 = r.get('crop_image')
                if crop_b64 and crop_b64.startswith('data:image'):
                    crop_data = base64.b64decode(crop_b64.split(',', 1)[1])
                    zf.writestr(f'{name_prefix}_crop.jpg', crop_data)

                for i, tc in enumerate(r.get('top_candidates', [])):
                    cb = tc.get('crop_base64')
                    if cb and cb.startswith('data:image'):
                        imgd = base64.b64decode(cb.split(',', 1)[1])
                        zf.writestr(f'{name_prefix}_candidate_{i+1}.jpg', imgd)

                per = {
                    'filename': name_prefix,
                    'bbox': r.get('bbox'),
                    'score': r.get('score'),
                    'sub_scores': r.get('sub_scores'),
                    'explanation': r.get('explanation'),
                    'explanation_full': r.get('explanation_full', ''),
                    'top_candidates': r.get('top_candidates', []),
                }
                zf.writestr(f'{name_prefix}_report.json', json.dumps(per, ensure_ascii=False, indent=2))

                lines = []
                lines.append(f'文件: {name_prefix}')
                lines.append(f"bbox: {per['bbox']}")
                lines.append(f"score: {per['score']}")
                lines.append('\n各项得分：')
                for k, v in (per['sub_scores'] or {}).items():
                    lines.append(f'- {k}: {v}')
                lines.append('\n简要说明：')
                lines.append(per.get('explanation') or '')
                lines.append('\n详细说明：')
                lines.append(per.get('explanation_full') or '')
                zf.writestr(f'{name_prefix}_report.txt', '\n'.join(lines))

                combined.append(per)

            zf.writestr('combined_report.json', json.dumps(combined, ensure_ascii=False, indent=2))
        zip_buffer.seek(0)
        return Response(zip_buffer.read(), mimetype='application/zip', headers={
            'Content-Disposition': 'attachment;filename=batch_report.zip'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/assistant', methods=['POST'])
def assistant_chat():
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    message = data.get('message', '')
    try:
        from src.utils import load_config
        from src.llm_crop_explainer import LLMCropExplainer

        config_path = get_config_path()
        cfg = load_config(config_path)
        llm_cfg = cfg.get('llm', {})
        llm_impl = None
        try:
            llm_impl = LLMCropExplainer(llm_cfg)
        except Exception:
            llm_impl = None

        if llm_impl and getattr(llm_impl, 'api_key', None):
            try:
                image_b64 = data.get('image_base64') if isinstance(data, dict) else None
                reply = llm_impl.chat(message, image_b64=image_b64)
                return jsonify({'reply': reply, 'llm_used': True})
            except Exception:
                import traceback
                traceback.print_exc()
                fallback = (
                    '抱歉，AI 助手暂时无法连接外部模型。你可以询问裁剪规则、'
                    '评分含义，或上传图片使用解释裁剪功能。'
                )
                return jsonify({'reply': fallback, 'llm_used': False})
        else:
            fallback = (
                'AI 助手未启用或未配置 API key。当前可以帮助解读分数含义；'
                '配置 LLM 后可以获得更详细的裁剪分析。'
            )
            return jsonify({'reply': fallback, 'llm_used': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/convert_clip', methods=['POST'])
def convert_clip():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data.get('text') or ''
    try:
        prompts = convert_to_clip_prompts(text)
        return jsonify({'prompts': prompts})
    except Exception as e:
        app.logger.exception('convert_clip failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/assistant_stream', methods=['POST'])
def assistant_stream():
    data = request.get_json(silent=True)
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    message = data.get('message', '')

    try:
        from src.utils import load_config
        from src.llm_crop_explainer import LLMCropExplainer

        config_path = get_config_path()
        cfg = load_config(config_path)
        llm_cfg = cfg.get('llm', {})
        try:
            llm_impl = LLMCropExplainer(llm_cfg)
        except Exception:
            llm_impl = None

        history = data.get('history') if isinstance(data, dict) else None

        def generate():
            if llm_impl and getattr(llm_impl, 'api_key', None):
                try:
                    image_b64 = data.get('image_base64') if isinstance(data, dict) else None
                    for chunk in llm_impl.chat_stream(message, history=history, image_b64=image_b64):
                        yield chunk
                        import time
                        time.sleep(0.07)  
                except Exception:
                    yield "event: error\ndata: LLM 调用失败\n\n"
            else:
                fallback = 'AI 助手未启用或未配置 API key。你可以先配置 LLM，或上传图片使用本地裁剪解释功能。'
                yield f"data: {fallback}\n\n"
            yield "event: done\ndata: \n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    try:
        config_path = get_config_path()
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
                img_data = base64.b64decode(crop_base64.split(",")[1])
                zip_file.writestr(f"{filename}_crop.jpg", img_data)
    zip_buffer.seek(0)
    return Response(
        zip_buffer.read(),
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment;filename=crops.zip"},
    )


@app.route('/api/export_report', methods=['POST'])
def export_report():
    data = request.get_json(silent=True)
    if not data or 'result' not in data:
        return jsonify({'error': 'No result provided'}), 400
    res = data['result']

    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            orig_b64 = res.get('original_image')
            if orig_b64 and orig_b64.startswith('data:image'):
                orig_data = base64.b64decode(orig_b64.split(',', 1)[1])
                zf.writestr('original.jpg', orig_data)

            crop_b64 = res.get('crop_image')
            if crop_b64 and crop_b64.startswith('data:image'):
                crop_data = base64.b64decode(crop_b64.split(',', 1)[1])
                zf.writestr('crop.jpg', crop_data)

            report = {
                'bbox': res.get('bbox'),
                'score': res.get('score'),
                'sub_scores': res.get('sub_scores'),
                'explanation': res.get('explanation'),
                'explanation_full': res.get('explanation_full', ''),
                'top_candidates': res.get('top_candidates', []),
            }
            zf.writestr('report.json', json.dumps(report, ensure_ascii=False, indent=2))

            text_lines = []
            text_lines.append('详细报告')
            text_lines.append('-------------------------')
            text_lines.append(f"bbox: {report['bbox']}")
            text_lines.append(f"score: {report['score']}")
            text_lines.append('\n各项得分：')
            for k, v in (report['sub_scores'] or {}).items():
                text_lines.append(f"- {k}: {v}")
            text_lines.append('\n简要说明：')
            text_lines.append(report.get('explanation') or '')
            text_lines.append('\n详细说明：')
            text_lines.append(report.get('explanation_full') or '')
            zf.writestr('report.txt', '\n'.join(text_lines))

        zip_buffer.seek(0)
        return Response(zip_buffer.read(), mimetype='application/zip', headers={
            'Content-Disposition': 'attachment;filename=report.zip'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_report_pdf', methods=['POST'])
def export_report_pdf():
    data = request.get_json(silent=True)
    if not data or 'result' not in data:
        return jsonify({'error': 'No result provided'}), 400
    res = data['result']
    font_path = data.get('font_path')

    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from PIL import Image
    except Exception as e:
        return jsonify({'error': 'Missing dependency: please install reportlab and pillow', 'detail': str(e)}), 500

    try:
        orig_b64 = res.get('original_image')
        crop_b64 = res.get('crop_image')

        def b64_to_pil(b64):
            if not b64 or not b64.startswith('data:image'):
                return None
            b = base64.b64decode(b64.split(',', 1)[1])
            return Image.open(io.BytesIO(b)).convert('RGB')

        orig_img = b64_to_pil(orig_b64)
        crop_img = b64_to_pil(crop_b64)

        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=A4)
        w, h = A4

        registered_font = None
        font_used_path = None
        if font_path:
            try:
                pdfmetrics.registerFont(TTFont('CustomFont', font_path))
                registered_font = 'CustomFont'
                font_used_path = font_path
            except Exception:
                registered_font = None
                font_used_path = None

        if not registered_font:
            try:
                fonts_dir = PROJECT_ROOT / 'fonts'
                if fonts_dir.exists() and fonts_dir.is_dir():
                    for f in sorted(fonts_dir.iterdir()):
                        if f.suffix.lower() in ('.ttf', '.ttc', '.otf'):
                            try:
                                pdfmetrics.registerFont(TTFont('CustomFont', str(f)))
                                registered_font = 'CustomFont'
                                font_used_path = str(f)
                                break
                            except Exception:
                                continue
            except Exception:
                pass

        if not registered_font:
            candidates = [
                os.path.expandvars(r'%SystemRoot%\\Fonts\\msyh.ttc'),
                os.path.expandvars(r'%SystemRoot%\\Fonts\\simsun.ttc'),
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/arphic/ukai.ttc'
            ]
            for p in candidates:
                try:
                    if p and os.path.exists(p):
                        pdfmetrics.registerFont(TTFont('CustomFont', p))
                        registered_font = 'CustomFont'
                        font_used_path = p
                        break
                except Exception:
                    continue

        title_font = registered_font or 'Helvetica'
        c.setFont(title_font, 18)
        c.drawString(40, h - 60, '裁剪详细报告')
        c.setFont(title_font, 10)
        c.drawString(40, h - 80, f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        y = h - 120
        img_max_w = (w - 120) / 2
        img_max_h = 200
        if orig_img:
            ir = ImageReader(orig_img)
            ow, oh = orig_img.size
            scale = min(img_max_w / ow, img_max_h / oh, 1.0)
            iw, ih = ow * scale, oh * scale
            c.drawImage(ir, 40, y - ih, width=iw, height=ih)
            c.drawString(40, y - ih - 12, '原图（含框）')
        if crop_img:
            ir2 = ImageReader(crop_img)
            cw2, ch2 = crop_img.size
            scale2 = min(img_max_w / cw2, img_max_h / ch2, 1.0)
            iw2, ih2 = cw2 * scale2, ch2 * scale2
            c.drawImage(ir2, 60 + img_max_w, y - ih2, width=iw2, height=ih2)
            c.drawString(60 + img_max_w, y - ih2 - 12, '裁剪图')

        text_y = y - img_max_h - 36
        if text_y < 120:
            c.showPage()
            text_y = h - 60

        c.setFont(title_font, 12)
        c.drawString(40, text_y, '裁剪信息')
        c.setFont(title_font, 10)
        text_y -= 18
        c.drawString(40, text_y, f"bbox: {res.get('bbox')}")
        text_y -= 14
        c.drawString(40, text_y, f"score: {res.get('score')}")
        text_y -= 18

        sub = res.get('sub_scores') or {}
        c.drawString(40, text_y, '各维度得分')
        text_y -= 14
        for k, v in sub.items():
            if text_y < 80:
                c.showPage(); text_y = h - 60
            c.drawString(60, text_y, f"- {k}: {v}")
            text_y -= 12

        if text_y < 140:
            c.showPage(); text_y = h - 60
        c.setFont(title_font, 12)
        c.drawString(40, text_y, '简要说明')
        c.setFont(title_font, 10)
        text_y -= 18
        brief = res.get('explanation') or ''
        def wrap_text_by_width(text, font_name, font_size, max_width):
            lines = []
            for para in str(text).split('\n'):
                if para == '':
                    lines.append('')
                    continue
                cur = ''
                for ch in para:
                    w = pdfmetrics.stringWidth(cur + ch, font_name, font_size)
                    if w <= max_width:
                        cur += ch
                    else:
                        if cur:
                            lines.append(cur)
                        cur = ch
                if cur:
                    lines.append(cur)
            return lines

        max_text_w = w - 80
        brief_lines = wrap_text_by_width(brief, title_font, 10, max_text_w)
        for line in brief_lines:
            if text_y < 80:
                c.showPage(); text_y = h - 60
                c.setFont(title_font, 10)
            c.drawString(40, text_y, line)
            text_y -= 12

        full = res.get('explanation_full') or ''
        if text_y < 140:
            c.showPage(); text_y = h - 60
        c.setFont(title_font, 12)
        c.drawString(40, text_y, '详细说明')
        c.setFont(title_font, 10)
        text_y -= 18
        full_lines = wrap_text_by_width(full, title_font, 10, max_text_w)
        for line in full_lines:
            if text_y < 80:
                c.showPage(); text_y = h - 60
                c.setFont(title_font, 10)
            c.drawString(40, text_y, line)
            text_y -= 12

        c.showPage()
        c.save()
        pdf_buf.seek(0)

        headers = {
            'Content-Disposition': 'attachment;filename=report.pdf',
        }
        if registered_font:
            headers['X-Font-Registered'] = 'true'
            try:
                safe_val = urllib.parse.quote(font_used_path or registered_font, safe='')
            except Exception:
                safe_val = os.path.basename(font_used_path or '') or registered_font
            headers['X-Font-Used'] = safe_val
            try:
                app.logger.info(f"PDF font registered: {font_used_path}")
            except Exception:
                pass
        else:
            headers['X-Font-Registered'] = 'false'
            headers['X-Font-Used'] = 'Helvetica'

        return Response(pdf_buf.read(), mimetype='application/pdf', headers=headers)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def main():
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
