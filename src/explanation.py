"""Template-based explanation generator for cropping results."""

from __future__ import annotations

from typing import Dict, List, Optional

from .utils import SubScores


class ExplanationGenerator:
    """Generate human-readable explanations for why a crop was selected."""

    def __init__(self, config: dict = None):
        """Initialize the explanation generator.

        Args:
            config: Optional configuration dict (reserved for future use).
        """
        pass

    # Templates keyed by which scores are dominant
    TEMPLATES = {
        "saliency_high": "显著性主体保留完整",
        "aesthetic_high": "美学评分较高",
        "thirds_good": "主体中心接近三分线位置",
        "subject_intact": "边界未明显切断主要目标",
        "composition_good": "构图平衡合理",
        "bright_sharp": "画面清晰明亮",
        "saliency_low": "显著性保留一般",
        "aesthetic_low": "美学评分中等",
        "subject_cut": "部分主体被边界截断",
    }

    def generate(
        self,
        sub_scores: SubScores,
        has_subject: bool = True,
    ) -> str:
        """Generate a template-based explanation for the crop selection.

        Args:
            sub_scores: The winning candidate's sub-scores (normalized).
            has_subject: Whether objects were detected.

        Returns:
            Explanation string (Chinese, 20-60 chars).
        """
        reasons = []

        # Check each dimension and add reasons
        if sub_scores.saliency >= 0.7:
            reasons.append(self.TEMPLATES["saliency_high"])
        elif sub_scores.saliency < 0.3:
            reasons.append(self.TEMPLATES["saliency_low"])

        if sub_scores.aesthetic >= 0.7:
            reasons.append(self.TEMPLATES["aesthetic_high"])
        elif sub_scores.aesthetic < 0.3:
            reasons.append(self.TEMPLATES["aesthetic_low"])

        if sub_scores.thirds >= 0.6:
            reasons.append(self.TEMPLATES["thirds_good"])

        if has_subject:
            if sub_scores.subject >= 0.7:
                reasons.append(self.TEMPLATES["subject_intact"])
            elif sub_scores.subject < 0.4:
                reasons.append(self.TEMPLATES["subject_cut"])

        if sub_scores.composition >= 0.7:
            reasons.append(self.TEMPLATES["composition_good"])

        if sub_scores.sharpness >= 0.7 and sub_scores.brightness >= 0.7:
            reasons.append(self.TEMPLATES["bright_sharp"])

        # Always include at least one reason
        if not reasons:
            reasons.append(self.TEMPLATES["aesthetic_high"])

        # Combine into explanation
        explanation = "选择该区域是因为" + "，".join(reasons) + "。"
        return explanation

    def generate_english(
        self,
        sub_scores: SubScores,
        has_subject: bool = True,
    ) -> str:
        """Generate an English explanation (alternative output)."""
        reasons = []

        if sub_scores.saliency >= 0.7:
            reasons.append("salient subject well preserved")
        if sub_scores.aesthetic >= 0.7:
            reasons.append("high aesthetic quality")
        if sub_scores.thirds >= 0.6:
            reasons.append("subject near rule-of-thirds position")
        if has_subject and sub_scores.subject >= 0.7:
            reasons.append("main subject intact")
        if sub_scores.composition >= 0.7:
            reasons.append("balanced composition")
        if sub_scores.sharpness >= 0.7 and sub_scores.brightness >= 0.7:
            reasons.append("clear and well-lit")

        if not reasons:
            reasons.append("best overall score")

        return "Selected because: " + ", ".join(reasons) + "."
