"""Template-based explanation generator for cropping results."""

from __future__ import annotations

from .utils import SubScores


class ExplanationGenerator:
    """Generate short human-readable explanations for why a crop was selected."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def generate(
        self,
        sub_scores: SubScores,
        has_subject: bool = True,
    ) -> str:
        """Generate a Chinese explanation, roughly 20-60 characters."""
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

        if sub_scores.area_prior >= 0.7:
            reasons.append("取景范围适中")

        if sub_scores.sharpness >= 0.7 and sub_scores.brightness >= 0.7:
            reasons.append("画面清晰明亮")

        if not reasons:
            reasons.append("综合评分最好")

        return "选择该区域是因为" + "，".join(reasons[:3]) + "。"

    def generate_english(
        self,
        sub_scores: SubScores,
        has_subject: bool = True,
    ) -> str:
        """Generate a short English explanation."""
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
        if sub_scores.area_prior >= 0.7:
            reasons.append("framing size is appropriate")
        if sub_scores.sharpness >= 0.7 and sub_scores.brightness >= 0.7:
            reasons.append("image is clear and well-lit")

        if not reasons:
            reasons.append("it has the best overall score")

        return "Selected because " + ", ".join(reasons[:3]) + "."
