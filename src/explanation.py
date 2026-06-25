from __future__ import annotations
from .utils import SubScores
from .llm_crop_explainer import LLMCropExplainer
import logging

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.llm_explainer = None
        llm_cfg = self.config.get("llm", {})
        enabled_flag = llm_cfg.get("enable_llm_reason", llm_cfg.get("enabled", False))
        if enabled_flag:
            self.llm_explainer = LLMCropExplainer(llm_cfg)
        aesthetic_cfg = self.config.get("models", {}).get("aesthetic", {})
        self.pos_prompts = aesthetic_cfg.get("positive_prompts", [])
        self.neg_prompts = aesthetic_cfg.get("negative_prompts", [])

    def _old_generate_short(self, sub_scores: SubScores, has_subject: bool = True) -> str:
        reasons = []
        if sub_scores.saliency >= 0.7:
            reasons.append("显著主体保留较完整")
        elif sub_scores.saliency < 0.3:
            reasons.append("显著性虽不突出但整体更均衡")
        if sub_scores.aesthetic >= 0.7:
            reasons.append("美学评分较高")
        elif sub_scores.aesthetic < 0.3:
            reasons.append("美学评分中等")
        if sub_scores.thirds >= 0.6:
            reasons.append("主体位置接近三分线")
        if has_subject:
            if sub_scores.subject >= 0.7:
                reasons.append("主要目标较完整")
            elif sub_scores.subject < 0.4:
                reasons.append("主体完整性一般")
        if sub_scores.composition >= 0.7:
            reasons.append("构图较平衡")
        if sub_scores.roi_discard >= 0.65:
            reasons.append("保留区域与舍弃区域区分清晰")
        if sub_scores.boundary_cut <= 0.25 and sub_scores.roi_saliency >= 0.5:
            reasons.append("边界未明显切断主体结构")
        if sub_scores.area_prior >= 0.7:
            reasons.append("取景范围适中")
        if sub_scores.sharpness >= 0.7 and sub_scores.brightness >= 0.7:
            reasons.append("画面清晰明亮")
        if not reasons:
            reasons.append("综合评分最好")
        return "选择该区域是因为" + "，".join(reasons[:3]) + "。"

    def generate(
        self,
        sub_scores: SubScores,
        has_subject: bool = True,
    ) -> str:
        """原始接口：仅返回短文案（兼容旧逻辑）"""
        return self._old_generate_short(sub_scores, has_subject)

    def generate_with_image(
        self,
        origin_image,
        crop_image,
        sub_scores: SubScores,
        detected_objects,
        has_subject: bool = True
    ) -> tuple[str, str]:

        if self.llm_explainer is None:
            short_text = self._old_generate_short(sub_scores, has_subject)
            return short_text, short_text
        try:
            short, full = self.llm_explainer.generate_reason(
                origin_img=origin_image,
                crop_img=crop_image,
                sub_scores=sub_scores,
                detected_objects=detected_objects,
                pos_prompts=self.pos_prompts,
                neg_prompts=self.neg_prompts
            )
            if not short or not full:
                raise ValueError("LLM返回文案为空")
            return short, full
        except Exception as e:
            logger.error(f"LLM生成理由降级至模板: {e}")
            fallback_text = self._old_generate_short(sub_scores, has_subject)
            return fallback_text, fallback_text

    def generate_english(
        self,
        sub_scores: SubScores,
        has_subject: bool = True,
    ) -> str:
        reasons = []
        if sub_scores.saliency >= 0.7:
            reasons.append("salient content is preserved")
        if sub_scores.aesthetic >= 0.7:
            reasons.append("aesthetic score is high")
        if sub_scores.thirds >= 0.6:
            reasons.append("subject is near a rule-of-thirds line")
        if has_subject and sub_scores.subject >= 0.7:
            reasons.append("main subject is intact")
        if sub_scores.composition >= 0.7:
            reasons.append("composition is balanced")
        if sub_scores.roi_discard >= 0.65:
            reasons.append("ROI and discarded region are well separated")
        if sub_scores.boundary_cut <= 0.25 and sub_scores.roi_saliency >= 0.5:
            reasons.append("crop boundaries avoid cutting salient structure")
        if sub_scores.area_prior >= 0.7:
            reasons.append("framing size is appropriate")
        if sub_scores.sharpness >= 0.7 and sub_scores.brightness >= 0.7:
            reasons.append("image is clear and well-lit")
        if not reasons:
            reasons.append("it has the best overall score")
        return "Selected because " + ", ".join(reasons[:3]) + "."