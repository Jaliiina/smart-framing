import base64
import cv2
import numpy as np
import json
import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    def load_dotenv():
        return None

try:
    from dashscope import MultiModalConversation
    import dashscope
except Exception:
    class MultiModalConversation:
        @staticmethod
        def call(*args, **kwargs):
            raise RuntimeError("dashscope SDK 未安装；请安装或在配置中关闭 llm.enabled")

    dashscope = None
logger = logging.getLogger(__name__)

class LLMCropExplainer:
    def __init__(self, llm_cfg):
        self.api_key = os.getenv("QWEN_API_KEY", llm_cfg.get("api_key", ""))
        self.model = os.getenv("QWEN_MODEL", llm_cfg.get("model", "qwen3.7-plus"))
        self.enable = llm_cfg.get("enable_llm_reason", llm_cfg.get("enabled", True))
        try:
            self.timeout = float(llm_cfg.get("timeout", os.getenv("QWEN_TIMEOUT", 8.0)))
        except Exception:
            self.timeout = 8.0
        dashscope.api_key = self.api_key
        self.assistant_system_prompt = llm_cfg.get(
            "assistant_system_prompt",
            "你是一个图像裁剪与机器视觉专家，擅长解释裁剪决策、指出画面问题，并给出改进建议。回答简洁清晰。不能使用markdown格式",
        )
        self.crop_system_prompt = llm_cfg.get(
            "crop_system_prompt",
            "你是专业摄影构图顾问，对比原图与系统裁剪图，结合下方打分数据输出两段文案：\n1. short：20~60字，前端卡片展示，简洁概括裁剪优势\n2. full：150~300字，导出报告详细分析，包含构图、主体、干扰物、画质、留白说明\n要求：\n- 不要堆砌分数，使用摄影专业术语；人像重点说明人脸/肢体无裁切；风景重点讲视觉重心、地平线；\n- 说明保留核心主体、剔除杂乱干扰、规避边缘裁切缺陷；\n- 差异化输出，禁止模板化套话；\n输出严格JSON格式，仅返回{\"short\":\"xxx\",\"full\":\"xxx\"}，无多余内容.",
        )
        self.clip_expert_system_prompt = llm_cfg.get(
            "clip_expert_system_prompt",
            (
                "你是一名 CLIP 提示词专家：从用户的中文输入中提取核心视觉要素，"
                "生成 5-8 个英文短语作为 CLIP 文本提示词。每个短语 2-6 个单词，简洁聚焦。"
                "只返回 JSON 数组，例如 [\"portrait smiling woman\", \"golden hour\"]，不要附加解释。"
                "\n\n"
                "提取要点：主体、风格、光线、色彩、构图、氛围、情绪"
            ),
        )

    @staticmethod
    def cv2_img_to_b64(img: np.ndarray, quality=85) -> str:
        success, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise RuntimeError("图片编码失败")
        b64_raw = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_raw}"

    def build_prompt(self, origin_img, crop_img, sub_scores, detected_objects, cfg_pos, cfg_neg):
        origin_b64 = self.cv2_img_to_b64(origin_img)
        crop_b64 = self.cv2_img_to_b64(crop_img)
        def _make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "__dict__"):
                result = {}
                for k, v in vars(obj).items():
                    result[k] = _make_serializable(v)
                return result
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            return obj

        score_json = json.dumps(_make_serializable(sub_scores), ensure_ascii=False, indent=2)
        obj_cls = [getattr(obj, "class_name", getattr(obj, "class_id", None)) for obj in detected_objects] if detected_objects else []
        try:
            logger.debug("LLM 请求 - sub_scores: %s", score_json)
            logger.debug("LLM 请求 - detected_objects: %s", obj_cls)
        except Exception:
            pass

        system_text = self.crop_system_prompt
        user_text = f"""
【多维打分】
{score_json}
【检测到物体类别ID】{obj_cls}
【模型正向美学标准】{cfg_pos}
【模型规避缺陷】{cfg_neg}
"""
        messages = [
            {"role": "system", "content": [{"text": system_text}]},
            {
                "role": "user",
                "content": [
                    {"image": origin_b64},
                    {"image": crop_b64},
                    {"text": user_text}
                ]
            }
        ]
        return messages

    def generate_reason(self, origin_img, crop_img, sub_scores, detected_objects, pos_prompts, neg_prompts):
        if not self.enable:
            raise ValueError("LLM 未启用（配置）")
        if not self.api_key:
            raise ValueError("LLM 未配置 API key（请在 .env 或环境变量中设置 QWEN_API_KEY 或在 llm.api_key 中填入）")

        messages = self.build_prompt(origin_img, crop_img, sub_scores, detected_objects, pos_prompts, neg_prompts)
        try:
            resp = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.warning(f"大模型调用失败（网络/SDK）：{e}")
            raise

        try:
            logger.debug(f"LLM 原始响应: {getattr(resp, '__dict__', str(resp))}")

            content = None
            if hasattr(resp, "output") and getattr(resp.output, "choices", None):
                try:
                    content = resp.output.choices[0].message.content
                except Exception:
                    content = None

            def _collect_texts(obj):
                texts = []
                if obj is None:
                    return texts
                if isinstance(obj, str):
                    texts.append(obj)
                    return texts
                if isinstance(obj, dict):
                    for v in obj.values():
                        texts.extend(_collect_texts(v))
                    return texts
                if isinstance(obj, (list, tuple)):
                    for it in obj:
                        texts.extend(_collect_texts(it))
                    return texts
                try:
                    texts.append(str(obj))
                except Exception:
                    pass
                return texts

            json_str = ""
            if isinstance(content, str):
                json_str = content
            else:
                texts = _collect_texts(content)
                if texts:
                    json_str = max(texts, key=lambda s: len(s))

            if not json_str and hasattr(resp, "text"):
                json_str = getattr(resp, "text")

            if not json_str:
                json_str = str(resp)

            json_str = (json_str or "").strip()

            parsed = None
            if json_str:
                try:
                    parsed = json.loads(json_str)
                except Exception:
                    import re
                    m = re.search(r"\{.*\}", json_str, re.S)
                    if m:
                        candidate = m.group(0)
                        try:
                            parsed = json.loads(candidate)
                        except Exception:
                            parsed = None

            if isinstance(parsed, dict):
                short = parsed.get("short", "")
                full = parsed.get("full", "")
                if not short and not full:
                    clean = json_str
                    return (clean[:200].strip(), clean)
                return short, full

            if json_str:
                clean = json_str
                short = clean if len(clean) <= 200 else clean[:200].rsplit("\n", 1)[0]
                full = clean
                return short, full

            raise ValueError("无法解析 LLM 返回的文本内容")
        except Exception as e:
            logger.warning(f"大模型生成裁剪理由解析失败: {str(e)} | 原始响应已记录到 debug")
            raise

    def convert_to_clip_prompts(self, user_text: str, n: int = 6) -> list:
        if not self.enable:
            raise RuntimeError("LLM 未启用（配置）")
        if not self.api_key:
            raise RuntimeError("LLM 未配置 API key")

        system_text = self.clip_expert_system_prompt
        user_content = {"text": f"Please convert: {user_text}. Return {n} concise prompts."}
        messages = [
            {"role": "system", "content": [{"text": system_text}]},
            {"role": "user", "content": [user_content]},
        ]

        try:
            resp = MultiModalConversation.call(model=self.model, messages=messages, timeout=self.timeout)
        except Exception as e:
            logger.warning(f"CLIP prompts LLM 调用失败：{e}")
            raise

        try:
            content = None
            if hasattr(resp, "output") and getattr(resp.output, "choices", None):
                try:
                    content = resp.output.choices[0].message.content
                except Exception:
                    content = None
            texts = []
            def _collect(obj):
                if obj is None:
                    return []
                if isinstance(obj, str):
                    return [obj]
                if isinstance(obj, dict):
                    res = []
                    for v in obj.values():
                        res += _collect(v)
                    return res
                if isinstance(obj, (list, tuple)):
                    res = []
                    for it in obj:
                        res += _collect(it)
                    return res
                try:
                    return [str(obj)]
                except Exception:
                    return []

            texts = _collect(content) if content is not None else []
            if not texts and hasattr(resp, "text"):
                texts = [getattr(resp, "text")]
            if texts:
                candidate = max(texts, key=lambda s: len(s))
                import json as _json, re as _re
                try:
                    parsed = _json.loads(candidate)
                    if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                        return parsed
                except Exception:
                    m = _re.search(r"\[.*\]", candidate, _re.S)
                    if m:
                        try:
                            parsed = _json.loads(m.group(0))
                            if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                                return parsed
                        except Exception:
                            pass
                    parts = [p.strip().strip('"') for p in _re.split(r"\n|\||;|,", candidate) if p.strip()]
                    parts = [p for p in parts if 1 <= len(p.split()) <= 8]
                    return parts[:n]

            raise RuntimeError("无法从 LLM 响应中提取 CLIP prompts")
        except Exception as e:
            logger.warning(f"解析 CLIP prompts 失败: {e}")
            raise

        

    def chat(self, user_message: str, history: list | None = None, image_b64: str | None = None) -> str:
        if not self.enable:
            raise RuntimeError("LLM 未启用（配置）")
        if not self.api_key:
            raise RuntimeError("LLM 未配置 API key")

        system_text = self.assistant_system_prompt
        user_content = []
        if image_b64:
            user_content.append({"image": image_b64})
        user_content.append({"text": user_message})

        messages = [
            {"role": "system", "content": [{"text": system_text}]},
            {"role": "user", "content": user_content},
        ]
        if history:
            for msg in history:
                if msg.get("role") and msg.get("text"):
                    messages.append({"role": msg["role"], "content": [{"text": msg["text"]}]})

        try:
            resp = MultiModalConversation.call(model=self.model, messages=messages, timeout=self.timeout)
        except Exception as e:
            logger.warning(f"LLM chat 调用失败：{e}")
            raise

        try:
            content = None
            if hasattr(resp, "output") and getattr(resp.output, "choices", None):
                try:
                    content = resp.output.choices[0].message.content
                except Exception:
                    content = None

            def _collect(obj):
                if obj is None:
                    return []
                if isinstance(obj, str):
                    return [obj]
                if isinstance(obj, dict):
                    res = []
                    for v in obj.values():
                        res += _collect(v)
                    return res
                if isinstance(obj, (list, tuple)):
                    res = []
                    for it in obj:
                        res += _collect(it)
                    return res
                try:
                    return [str(obj)]
                except Exception:
                    return []

            texts = _collect(content) if content is not None else []
            if not texts and hasattr(resp, "text"):
                texts = [getattr(resp, "text")]
            if texts:
                return max(texts, key=lambda s: len(s))
            return str(resp)
        except Exception as e:
            logger.warning(f"解析 LLM chat 响应失败: {e}")
            raise

    def chat_stream(self, user_message: str, history: list | None = None, image_b64: str | None = None, chunk_size: int = 120):
        import time
        try:
            full = self.chat(user_message, history=history, image_b64=image_b64)
        except Exception as e:
            logger.warning(f"chat同步调用失败: {e}")
            yield "event: error\ndata: AI模型调用失败\n\n"
            yield "event: done\ndata: \n\n"
            return

        if not full:
            yield "event: done\ndata: \n\n"
            return

        import re
        sentence_end_re = re.compile(r'(.*?([。！？!?;；]|\.|\?|!))(\s+|$)', re.S)
        sentences = []
        pos = 0
        for m in sentence_end_re.finditer(full):
            sentences.append(m.group(1).strip())
            pos = m.end()
        if pos < len(full):
            rest = full[pos:].strip()
            if rest:
                sentences.append(rest)
        if not sentences:
            for i in range(0, len(full), chunk_size):
                yield f"data: {full[i:i+chunk_size]}\n\n"
                time.sleep(0.06)
            yield "event: done\ndata: \n\n"
            return

        current = ''
        for s in sentences:
            if not current:
                current = s
            elif len(current) + len(s) + 1 <= chunk_size:
                current = current + ' ' + s
            else:
                yield f"data: {current}\n\n"
                time.sleep(0.08)  
                current = s
        if current:
            yield f"data: {current}\n\n"
        yield "event: done\ndata: \n\n"