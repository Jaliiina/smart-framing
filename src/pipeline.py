"""Top-level AestheticCropper pipeline orchestrator."""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix ultralytics settings permission issue
_yolo_settings_dir = str(Path.home() / ".config" / "ultralytics")
os.makedirs(_yolo_settings_dir, exist_ok=True)
os.environ.setdefault("YOLO_SETTINGS_DIR", _yolo_settings_dir)

import cv2
import numpy as np

from .utils import (
    BBox,
    CandidateResult,
    CropResult,
    DetectedObject,
    SubScores,
    draw_bbox,
    draw_multiple_bboxes,
    load_config,
    load_image,
    save_image,
)

logger = logging.getLogger(__name__)


class AestheticCropper:
    """Main pipeline: generate candidates, score, fuse, and output results."""

    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None):
        """Initialize the AestheticCropper pipeline.

        Args:
            config_path: Path to config.yaml.
            config: Optional pre-loaded config dict (overrides config_path).
        """
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Initialize all modules (lazy model loading)
        from .candidate_generation import CandidateGenerator
        from .saliency_detector import SaliencyDetector
        from .aesthetic_scorer import AestheticScorer
        from .subject_detector import SubjectDetector
        from .composition_scorer import CompositionScorer
        from .technical_quality import TechnicalQualityScorer
        from .fusion import FusionModule
        from .explanation import ExplanationGenerator
        from .reranker import LearnedReranker
        from .roi_discard_scorer import RoiDiscardScorer
        from .semantic_crop_scorer import SemanticCropScorer
        from .semantic_heatmap_scorer import SemanticHeatmapScorer
        from .scientific_optimizer import ScientificCropOptimizer
        from .subjectness_scorer import SubjectnessScorer

        self.candidate_gen = CandidateGenerator(self.config)
        self.saliency_det = SaliencyDetector(self.config)
        self.aesthetic_scorer = AestheticScorer(self.config)
        self.subject_det = SubjectDetector(self.config)
        self.comp_scorer = CompositionScorer(self.config)
        self.tech_scorer = TechnicalQualityScorer(self.config)
        self.semantic_crop_scorer = SemanticCropScorer(self.config)
        self.semantic_heatmap_scorer = SemanticHeatmapScorer(self.config)
        self.subjectness_scorer = SubjectnessScorer(self.config)
        self.roi_discard_scorer = RoiDiscardScorer(self.config)
        self.fusion = FusionModule(self.config)
        self.explainer = ExplanationGenerator(self.config)
        self.scientific_optimizer = ScientificCropOptimizer(
            self.config, self.roi_discard_scorer, self.semantic_crop_scorer
        )
        self.reranker = None
        reranker_cfg = self.config.get("reranker", {})
        if reranker_cfg.get("enabled", False):
            model_path = reranker_cfg.get("model_path", "models/testa_reranker.json")
            try:
                self.reranker = LearnedReranker.from_file(
                    model_path,
                    blend_with_fusion=reranker_cfg.get("blend_with_fusion", 0.0),
                    takeover_margin=reranker_cfg.get("takeover_margin", 0.04),
                    protect_high_quality_fusion=reranker_cfg.get(
                        "protect_high_quality_fusion", True
                    ),
                    protect_fusion_score_threshold=reranker_cfg.get(
                        "protect_fusion_score_threshold", 0.80
                    ),
                    large_area_takeover_threshold=reranker_cfg.get(
                        "large_area_takeover_threshold", 0.50
                    ),
                    blank_penalty_weight=reranker_cfg.get("blank_penalty_weight", 0.45),
                    artifact_penalty_weight=reranker_cfg.get("artifact_penalty_weight", 0.45),
                    saturated_penalty_weight=reranker_cfg.get("saturated_penalty_weight", 0.50),
                    blank_penalty_threshold=reranker_cfg.get("blank_penalty_threshold", 0.45),
                    artifact_penalty_threshold=reranker_cfg.get("artifact_penalty_threshold", 0.45),
                    saturated_penalty_threshold=reranker_cfg.get("saturated_penalty_threshold", 0.35),
                )
                logger.info(f"Learned reranker loaded from {model_path}")
            except Exception as exc:
                logger.warning(f"Failed to load learned reranker: {exc}")

        # Print model info
        u2net_cfg = self.config.get("models", {}).get("u2net", {})
        yolo_cfg = self.config.get("models", {}).get("yolo", {})
        aesthetic_cfg = self.config.get("models", {}).get("aesthetic", {})
        u2net_path = u2net_cfg.get("weights_path" if not u2net_cfg.get("use_lite", False) else "lite_weights_path", "models/u2netp.pth")
        u2net_name = "U2Net" if not u2net_cfg.get("use_lite", False) else "U2NetP (lite)"
        yolo_model = yolo_cfg.get("model_name", "yolov8n.pt")
        yolo_conf = yolo_cfg.get("confidence_threshold", 0.3)
        aesthetic_device = aesthetic_cfg.get("device", "cuda")
        u2net_device = u2net_cfg.get("device", "cuda")
        yolo_device = yolo_cfg.get("device", "cuda")
        aesthetic_path = aesthetic_cfg.get("model_path", "models/aesthetic_predictor.pth")
        clip_model = aesthetic_cfg.get("clip_model", "ViT-L/14")
        if aesthetic_cfg.get("use_laion_predictor", False):
            aesthetic_name = f"LAION aesthetic predictor ({clip_model}) [{aesthetic_path}]"
        elif aesthetic_cfg.get("use_clip_prompt_fallback", False):
            aesthetic_name = f"CLIP prompt scoring ({clip_model})"
        else:
            aesthetic_name = "hand-crafted fallback features"
        print(
            f"[Model Info] saliency={u2net_name} [{u2net_path}]; "
            f"subject={yolo_model} (conf={yolo_conf}); "
            f"aesthetic={aesthetic_name}; "
            f"u2net_device={u2net_device}, yolo_device={yolo_device}, aesthetic_device={aesthetic_device}"
        )

    def process(self, image_path: str) -> CropResult:
        """Process a single image through the full pipeline.

        Steps:
          0. Intent classification (multimodal LLM) → choose strategy
          1. Run U2-Net once → saliency map
          2. Run YOLOv8 once → detected objects
          3. Generate candidates (grid + saliency-guided)
          4. Score each candidate
          5. Fuse scores and select best
          6. Generate explanation

        Args:
            image_path: Path to input image.

        Returns:
            CropResult with best bbox, crop, scores, explanation, etc.
        """
        start_time = time.time()

        image = load_image(image_path)
        h, w = image.shape[:2]
        # 初始化兜底，解决best_crop未定义报错
        best_crop = image.copy()
        best_bbox = (0, 0, w, h)

        # --- Step 1: Run dual saliency (U2-Net + fallback) ---
        saliency_map, fallback_sal_map, is_uniform, fallback_uniform = (
            self.saliency_det.detect_dual(image)
        )

        # --- Step 2: Run YOLOv8 once → detected objects ---
        detected_objects = self.subject_det.detect(image, saliency_map=saliency_map)
        has_subject = len(detected_objects) > 0

        # --- Step 3: Generate candidates (grid + saliency-guided) ---
        candidates = self.candidate_gen.generate(
            image, saliency_map, detected_objects=detected_objects
        )
        logger.info(f"Generated {len(candidates)} candidates for {image_path}")

        if len(candidates) == 0:
            candidates = [(0, 0, w, h)]

        # Compute per-candidate scores for both saliency maps
        u2net_saliency_scores = self.saliency_det.score_candidates(
            saliency_map, candidates, image.shape
        )
        fallback_saliency_scores = (
            self.saliency_det.score_candidates(
                fallback_sal_map, candidates, image.shape
            )
            if fallback_sal_map is not saliency_map
            else u2net_saliency_scores
        )

        # --- Step 4: Score each candidate ---
        # 4a. Aesthetic scores
        aesthetic_scores = self.aesthetic_scorer.score_candidates(image, candidates)

        # 4b. Saliency preservation scores (primary map)
        saliency_scores = u2net_saliency_scores

        # 4b-alt. Dual saliency agreement scores (fallback map, used in fusion)
        dual_saliency_scores = fallback_saliency_scores

        # 4c. Composition scores
        composition_scores = self.comp_scorer.score_candidates(
            image, candidates, saliency_map, detected_objects
        )

        # 4d. Subject completeness scores
        subject_scores = self.subject_det.score_candidates(
            candidates, detected_objects, image.shape
        )
        has_subject = any(score is not None for score in subject_scores)
        subject_score_mode = getattr(self.subject_det, "last_score_mode", "subject")

        # 4e. Technical quality scores
        technical_scores = self.tech_scorer.score_candidates(image, candidates)

        # 4f. Semantic subjectness and distractor-aware ROI/discard scores
        semantic_heatmaps = self.semantic_heatmap_scorer.build_heatmaps(image)
        subjectness_maps = self.subjectness_scorer.build_maps(
            image=image,
            saliency_map=saliency_map,
            detected_objects=detected_objects,
            semantic_heatmaps=semantic_heatmaps,
        )
        subjectness_scores = self.subjectness_scorer.score_candidates(
            candidates, subjectness_maps
        )
        semantic_scores = self.semantic_crop_scorer.score_candidates(image, candidates)
        roi_discard_scores = self.roi_discard_scorer.score_candidates(
            image=image,
            bboxes=candidates,
            saliency_map=saliency_map,
            detected_objects=detected_objects,
            subjectness_maps=subjectness_maps,
            semantic_scores=semantic_scores,
        )

        # --- Step 5: Fuse scores and select best ---
        best, top_k, all_ranked = self.fusion.fuse(
            bboxes=candidates,
            aesthetic_scores=aesthetic_scores,
            saliency_scores=saliency_scores,
            composition_scores=composition_scores,
            subject_scores=subject_scores,
            technical_scores=technical_scores,
            roi_discard_scores=roi_discard_scores,
            semantic_scores=semantic_scores,
            subjectness_scores=subjectness_scores,
            saliency_is_uniform=is_uniform,
            has_subject=has_subject,
            image_shape=image.shape[:2],
            saliency_map=saliency_map,
            return_all=True,  # Get all ranked candidates for evaluation
            subject_source=subject_score_mode,
            dual_saliency_scores=dual_saliency_scores,
        )
        
        # Use all_ranked if available, otherwise fall back to candidates
        all_candidates = all_ranked if all_ranked else candidates

        if self.reranker is not None and all_ranked:
            all_candidates = self.reranker.rerank(all_ranked, image.shape[:2])
            best = all_candidates[0]
            top_k = all_candidates[: self.fusion.top_k_display]

        all_candidates = self.scientific_optimizer.optimize(
            image=image,
            ranked=all_candidates if all_candidates else top_k,
            detected_objects=detected_objects,
            saliency_map=saliency_map,
            subjectness_maps=subjectness_maps,
        )

        all_candidates = self._apply_final_quality_rerank(
            image=image,
            candidates=all_candidates,
        )
        all_candidates = self._apply_output_calibration(
            all_candidates,
            image_shape=image.shape[:2],
        )
        # 安全读取最优候选，边界保护
        if len(all_candidates) > 0:
            best = all_candidates[0]
            top_k = all_candidates[: self.fusion.top_k_display]
            x1, y1, x2, y2 = best.bbox
            # 裁剪坐标防越界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            best_crop = image[y1:y2, x1:x2]

        # --- Step 6: Generate explanation ---
        explanation_short, explanation_full = self.explainer.generate_with_image(
            origin_image=image,
            crop_image=best_crop,
            sub_scores=best.sub_scores,
            detected_objects=detected_objects,
            has_subject=has_subject
        )

        # --- Step 7: Build result ---
        elapsed = time.time() - start_time
        logger.info(
            f"Processed {image_path} in {elapsed:.2f}s | "
            f"best_score={best.final_score:.4f} bbox={best.bbox}"
        )

        return CropResult(
            image_path=image_path,
            best_bbox=best.bbox,
            best_crop=best_crop,
            best_score=best.final_score,
            best_sub_scores=best.sub_scores,
            top_candidates=top_k,
            explanation=explanation_short,  # 兼容旧字段，存短文案
            explanation_full=explanation_full, # 新增长报告文案
            saliency_map=saliency_map,
            detected_objects=detected_objects,
            all_candidates=all_candidates,
        )

    def _apply_final_quality_rerank(
        self,
        image: np.ndarray,
        candidates: List[CandidateResult],
    ) -> List[CandidateResult]:
        """Rerank near-final candidates with generic visual quality signals."""
        cfg = self.config.get("final_quality_rerank", {})
        if not cfg.get("enabled", False) or len(candidates) <= 1:
            return candidates

        top_n = max(2, int(cfg.get("top_n", 24)))
        seed_n = max(1, int(cfg.get("local_variant_seed_n", 6)))
        takeover_margin = float(cfg.get("takeover_margin", 0.035))
        base_weight = float(cfg.get("base_weight", 0.52))
        quality_weight = float(cfg.get("quality_weight", 0.48))
        variant_base_decay = float(cfg.get("variant_base_decay", 0.97))
        no_subject_max_score_drop = float(cfg.get("no_subject_max_score_drop", 0.02))
        pareto_weight = float(cfg.get("pareto_weight", 0.18))
        pool = list(candidates[: min(top_n, len(candidates))])
        tail = list(candidates[len(pool):])
        seen = {cand.bbox for cand in pool}
        h, w = image.shape[:2]
        for seed in list(pool[: min(seed_n, len(pool))]):
            for bbox in self._local_quality_variants(seed.bbox, h, w):
                if bbox in seen:
                    continue
                seen.add(bbox)
                pool.append(
                    CandidateResult(
                        bbox=bbox,
                        final_score=float(seed.final_score * variant_base_decay),
                        sub_scores=seed.sub_scores,
                    )
                )
            centered = self._content_centered_variant(image, seed.bbox)
            if centered is not None and centered not in seen:
                seen.add(centered)
                pool.append(
                    CandidateResult(
                        bbox=centered,
                        final_score=float(seed.final_score * variant_base_decay),
                        sub_scores=seed.sub_scores,
                    )
                )
            # paper_line = self._paper_line_art_variant(image, seed.bbox)
            # if paper_line is not None and paper_line not in seen:
            #     seen.add(paper_line)
            #     pool.append(
            #         CandidateResult(
            #             bbox=paper_line,
            #             final_score=float(seed.final_score * variant_base_decay),
            #             sub_scores=seed.sub_scores,
            #         )
            #     )

        scored = []
        original_best = pool[0]
        for cand in pool:
            quality, axes = self._generic_crop_quality(image, cand)
            pareto = self._pareto_balance_score(axes)
            adjusted = (
                base_weight * cand.final_score
                + quality_weight * quality
                + pareto_weight * pareto
            ) / max(1e-9, base_weight + quality_weight + pareto_weight)
            scored.append((adjusted, quality, cand))

        scored.sort(key=lambda item: item[0], reverse=True)
        picked = scored[0][2]
        best_adjusted = scored[0][0]
        original_adjusted = next(
            adjusted for adjusted, _quality, cand in scored if cand is original_best
        )
        if picked is not original_best and best_adjusted < original_adjusted + takeover_margin:
            picked = original_best
        if (
            picked is not original_best
            and original_best.sub_scores.subject < 0.20
            and picked.final_score < original_best.final_score - no_subject_max_score_drop
        ):
            picked = original_best

        reranked = [picked]
        for _adjusted, _quality, cand in scored:
            if cand is not picked:
                reranked.append(cand)
        return reranked + tail

    @staticmethod
    def _local_quality_variants(
        bbox: BBox,
        h: int,
        w: int,
    ) -> List[BBox]:
        x1, y1, x2, y2 = bbox
        bw, bh = max(8, x2 - x1), max(8, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        variants: List[BBox] = []
        shifts = [(-0.08, 0.0), (0.08, 0.0), (0.0, -0.08), (0.0, 0.08)]
        scales = [0.92]
        for dx, dy in shifts:
            nb = (
                int(round(x1 + dx * bw)),
                int(round(y1 + dy * bh)),
                int(round(x2 + dx * bw)),
                int(round(y2 + dy * bh)),
            )
            variants.append(_clamped_min_size_bbox(nb, h, w))
        for scale in scales:
            nw, nh = bw * scale, bh * scale
            nb = (
                int(round(cx - nw / 2)),
                int(round(cy - nh / 2)),
                int(round(cx + nw / 2)),
                int(round(cy + nh / 2)),
            )
            variants.append(_clamped_min_size_bbox(nb, h, w))
        return [b for b in variants if b[2] - b[0] >= 8 and b[3] - b[1] >= 8]

    @staticmethod
    def _content_centered_variant(
        image: np.ndarray,
        bbox: BBox,
    ) -> Optional[BBox]:
        x1, y1, x2, y2 = bbox
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None
        h, w = image.shape[:2]
        ch, cw = crop.shape[:2]
        if ch < 12 or cw < 12:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        edges = cv2.GaussianBlur(cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0, (0, 0), 1.2)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        activity = 0.55 * edges + 0.30 * sat + 0.15 * (1.0 - np.abs(val - 0.55))
        total = float(activity.sum())
        if total < 1e-6:
            return None
        ys, xs = np.mgrid[0:ch, 0:cw]
        cx_local = float((xs * activity).sum() / total) / max(1, cw)
        cy_local = float((ys * activity).sum() / total) / max(1, ch)
        dx = np.clip((cx_local - 0.5) * 0.55, -0.10, 0.10)
        dy = np.clip((cy_local - 0.5) * 0.45, -0.08, 0.08)
        if abs(dx) < 0.025 and abs(dy) < 0.025:
            return None
        bw, bh = x2 - x1, y2 - y1
        nb = (
            int(round(x1 + dx * bw)),
            int(round(y1 + dy * bh)),
            int(round(x2 + dx * bw)),
            int(round(y2 + dy * bh)),
        )
        return _clamped_min_size_bbox(nb, h, w)

    # @staticmethod
    # def _paper_line_art_variant(
    #     image: np.ndarray,
    #     bbox: BBox,
    # ) -> Optional[BBox]:
    #     """Tighten high-key paper/line-art crops around the main drawn structure."""
    #     x1, y1, x2, y2 = bbox
    #     crop = image[max(0, y1):y2, max(0, x1):x2]
    #     if crop.size == 0:
    #         return None
    #     h, w = image.shape[:2]
    #     ch, cw = crop.shape[:2]
    #     if ch < 48 or cw < 48:
    #         return None

    #     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    #     sat = hsv[:, :, 1] / 255.0
    #     val = hsv[:, :, 2] / 255.0
    #     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.GaussianBlur(
    #         cv2.Canny(gray, 55, 150).astype(np.float32) / 255.0,
    #         (0, 0),
    #         1.0,
    #     )
    #     paper = (sat < 0.22) & (val > 0.66)
    #     if float(paper.mean()) < 0.46 or float(sat.mean()) > 0.16 or float(val.mean()) < 0.58:
    #         return None

    #     structure = (val < 0.58) & (sat < 0.30)
    #     structure = cv2.morphologyEx(
    #         structure.astype(np.uint8),
    #         cv2.MORPH_OPEN,
    #         np.ones((3, 3), dtype=np.uint8),
    #     )
    #     n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
    #         structure,
    #         8,
    #     )
    #     if n_labels <= 1:
    #         return None

    #     crop_area = max(1, ch * cw)
    #     components = []
    #     for label in range(1, n_labels):
    #         area = int(stats[label, cv2.CC_STAT_AREA])
    #         if area < max(80, int(0.008 * crop_area)):
    #             continue
    #         lx = int(stats[label, cv2.CC_STAT_LEFT])
    #         ly = int(stats[label, cv2.CC_STAT_TOP])
    #         lw = int(stats[label, cv2.CC_STAT_WIDTH])
    #         lh = int(stats[label, cv2.CC_STAT_HEIGHT])
    #         area_ratio = area / crop_area
    #         box_ratio = (lw * lh) / crop_area
    #         if area_ratio > 0.42 or box_ratio > 0.88:
    #             continue
    #         components.append((area, lx, ly, lx + lw - 1, ly + lh - 1))
    #     if not components:
    #         return None

    #     components.sort(reverse=True)
    #     _area, ax1, ay1, ax2, ay2 = components[0]
    #     aw, ah = ax2 - ax1 + 1, ay2 - ay1 + 1
    #     if aw < 0.22 * cw or ah < 0.30 * ch:
    #         return None

    #     pad_x = int(round(max(12, 0.18 * aw)))
    #     pad_top = int(round(max(12, 0.32 * ah)))
    #     pad_bottom = int(round(max(10, 0.12 * ah)))
    #     nx1 = x1 + max(0, ax1 - pad_x)
    #     ny1 = y1 + max(0, ay1 - pad_top)
    #     nx2 = x1 + min(cw, ax2 + pad_x)
    #     ny2 = y1 + min(ch, ay2 + pad_bottom)

    #     old_area = max(1, (x2 - x1) * (y2 - y1))
    #     new_area = max(1, (nx2 - nx1) * (ny2 - ny1))
    #     if new_area > old_area * 0.88 or new_area < old_area * 0.30:
    #         return None
    #     return _clamped_min_size_bbox((nx1, ny1, nx2, ny2), h, w)

    def _generic_crop_quality(
        self,
        image: np.ndarray,
        candidate: CandidateResult,
    ) -> Tuple[float, Dict[str, float]]:
        """Score cleanliness, completeness, and boundary safety for a crop."""
        sub = candidate.sub_scores
        x1, y1, x2, y2 = candidate.bbox
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.0, {}
        h, w = image.shape[:2]
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, h * w)
        if 0.24 <= area_ratio <= 0.48:
            area_score = 1.0
        elif area_ratio < 0.24:
            area_score = max(0.0, 1.0 - (0.24 - area_ratio) / 0.20)
        else:
            area_score = max(0.0, 1.0 - (area_ratio - 0.48) / 0.28)

        artifact = float(
            np.clip(
                0.45 * sub.visual_artifact_penalty
                + 0.25 * sub.blank_area_penalty
                + 0.20 * sub.small_saturated_object_penalty
                + 0.10 * max(sub.distractor_map_score, sub.distractor_penalty),
                0.0,
                1.0,
            )
        )
        boundary_clean = 1.0 - float(np.clip(sub.boundary_cut, 0.0, 1.0))
        border_residue = self._border_residue_penalty(crop)
        # saturated_edge_residue = self._saturated_edge_residue_penalty(crop)
        cut_risk = self._structure_cut_penalty(crop)
        foreground_residue = self._foreground_residue_penalty(crop)
        # vertical_position = self._vertical_position_penalty(candidate.bbox, image.shape[:2])
        # paper_margin = self._paper_margin_penalty(crop)
        completeness = max(sub.subject, sub.subjectness)
        artifact_clean = 1.0 - max(artifact, 0.35 * foreground_residue)

        content_terms = (
            0.18 * sub.aesthetic
            + 0.17 * sub.composition
            + 0.12 * sub.saliency
            + 0.14 * sub.roi_discard
            + 0.12 * sub.semantic_score
            + 0.11 * completeness
            + 0.08 * boundary_clean
            + 0.05 * artifact_clean
            + 0.03 * area_score
        )
        large_area_penalty = max(0.0, (area_ratio - 0.56) / 0.24)
        penalty = (
            0.15 * border_residue
            + 0.11 * cut_risk
            + 0.04 * foreground_residue
            # + 0.16 * saturated_edge_residue
            # + 0.10 * vertical_position
            # + 0.18 * paper_margin
            + 0.14 * large_area_penalty
        )
        axes = {
            "content": float(np.clip(content_terms, 0.0, 1.0)),
            "cleanliness": float(np.clip(artifact_clean, 0.0, 1.0)),
            "boundary": float(np.clip(boundary_clean * (1.0 - cut_risk), 0.0, 1.0)),
            "completeness": float(np.clip(completeness, 0.0, 1.0)),
            "area": float(np.clip(area_score, 0.0, 1.0)),
            "residue": float(np.clip(1.0 - max(border_residue, foreground_residue), 0.0, 1.0)),
            # "residue": float(
            #     np.clip(
            #         1.0
            #         - max(
            #             border_residue,
            #             foreground_residue,
            #             saturated_edge_residue,
            #             paper_margin,
            #         ),
            #         0.0,
            #         1.0,
            #     )
            # ),
        }
        return float(np.clip(content_terms - penalty, 0.0, 1.0)), axes

    @staticmethod
    # def _vertical_position_penalty(bbox: BBox, image_shape: Tuple[int, int]) -> float:
    #     h, w = image_shape
    #     x1, y1, x2, y2 = bbox
    #     bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    #     cy = ((y1 + y2) / 2.0) / max(1, h)
    #     area_ratio = (bw * bh) / max(1, h * w)
    #     if area_ratio > 0.50:
    #         return 0.0
    #     return float(np.clip((cy - 0.62) / 0.18, 0.0, 1.0))

    # @staticmethod
    # def _paper_margin_penalty(crop: np.ndarray) -> float:
    #     h, w = crop.shape[:2]
    #     if h < 48 or w < 48:
    #         return 0.0
    #     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    #     sat = hsv[:, :, 1] / 255.0
    #     val = hsv[:, :, 2] / 255.0
    #     if float(sat.mean()) > 0.16 or float(val.mean()) < 0.56:
    #         return 0.0
    #     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.GaussianBlur(
    #         cv2.Canny(gray, 55, 150).astype(np.float32) / 255.0,
    #         (0, 0),
    #         1.0,
    #     )
    #     blank = (sat < 0.18) & (val > 0.66) & (edges < 0.04)
    #     if float(blank.mean()) < 0.38:
    #         return 0.0
    #     sx = max(6, int(round(w * 0.33)))
    #     sy = max(6, int(round(h * 0.16)))
    #     left_blank = float(blank[:, :sx].mean())
    #     bottom_blank = float(blank[-sy:, :].mean())
    #     return float(
    #         np.clip(
    #             0.55 * left_blank + 0.35 * float(blank.mean()) + 0.10 * bottom_blank - 0.55,
    #             0.0,
    #             1.0,
    #         )
    #     )

    # @staticmethod
    def _pareto_balance_score(axes: Dict[str, float]) -> float:
        if not axes:
            return 0.0
        values = np.array(list(axes.values()), dtype=np.float64)
        floor = float(values.min())
        mean = float(values.mean())
        return float(np.clip(0.62 * floor + 0.38 * mean, 0.0, 1.0))

    @staticmethod
    def _border_residue_penalty(crop: np.ndarray) -> float:
        """Detect visually distracting partial objects close to crop borders."""
        h, w = crop.shape[:2]
        if h < 12 or w < 12:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 170).astype(np.float32) / 255.0
        strip = max(4, int(min(h, w) * 0.075))
        border = np.zeros((h, w), dtype=bool)
        border[:strip, :] = True
        border[-strip:, :] = True
        border[:, :strip] = True
        border[:, -strip:] = True
        saturated = (sat > 0.52) & (val > 0.28)
        dark_chunk = (val < 0.20) & (edges > 0.03)
        edge_chunk = edges > 0.18
        residue = border & (saturated | dark_chunk | edge_chunk)
        return float(np.clip(residue.mean() / 0.055, 0.0, 1.0))

    @staticmethod
    # def _saturated_edge_residue_penalty(crop: np.ndarray) -> float:
    #     """Penalize clipped high-saturation color blobs at crop edges."""
    #     h, w = crop.shape[:2]
    #     if h < 16 or w < 16:
    #         return 0.0
    #     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    #     sat = hsv[:, :, 1] / 255.0
    #     val = hsv[:, :, 2] / 255.0
    #     hue = hsv[:, :, 0]
    #     strip = max(4, int(min(h, w) * 0.08))
    #     border = np.zeros((h, w), dtype=bool)
    #     border[:strip, :] = True
    #     border[-strip:, :] = True
    #     border[:, :strip] = True
    #     border[:, -strip:] = True
    #     warm_or_vivid = ((hue < 25) | (hue > 165) | (sat > 0.72))
    #     mask = (sat > 0.50) & (val > 0.28) & warm_or_vivid
    #     edge_mask = (mask & border).astype(np.uint8)
    #     if int(edge_mask.sum()) == 0:
    #         return 0.0
    #     n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
    #         edge_mask,
    #         8,
    #     )
    #     crop_area = max(1, h * w)
    #     penalty = 0.0
    #     for label in range(1, n_labels):
    #         area = int(stats[label, cv2.CC_STAT_AREA])
    #         area_ratio = area / crop_area
    #         if area_ratio < 0.001:
    #             continue
    #         penalty = max(penalty, float(np.clip(area_ratio / 0.035, 0.0, 1.0)))
    #     return penalty

    # @staticmethod
    def _foreground_residue_penalty(crop: np.ndarray) -> float:
        """Penalize small distracting blobs in lower/side foreground regions."""
        h, w = crop.shape[:2]
        if h < 16 or w < 16:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 170).astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:h, 0:w]
        yn = yy / max(1, h - 1)
        xn = xx / max(1, w - 1)
        lower = np.clip((yn - 0.45) / 0.45, 0.0, 1.0)
        side = np.clip(np.abs(xn - 0.5) * 2.0, 0.0, 1.0)
        foreground_prior = 0.62 * lower + 0.38 * side

        saturated = (sat > 0.55) & (val > 0.32)
        dark_object = (val < 0.23) & (edges > 0.03)
        mask = (saturated | dark_object).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        crop_area = max(1, h * w)
        penalty = 0.0
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            area_ratio = area / crop_area
            if area_ratio < 0.002 or area_ratio > 0.28:
                continue
            component = labels == label
            prior = float(foreground_prior[component].mean())
            edge_density = float(edges[component].mean()) if component.any() else 0.0
            area_score = float(np.clip((area_ratio - 0.002) / 0.050, 0.0, 1.0))
            penalty = max(penalty, area_score * (0.75 * prior + 0.25 * edge_density))
        return float(np.clip(penalty, 0.0, 1.0))

    @staticmethod
    def _structure_cut_penalty(crop: np.ndarray) -> float:
        """Penalize strong structures that terminate exactly on crop edges."""
        h, w = crop.shape[:2]
        if h < 12 or w < 12:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 170).astype(np.float32) / 255.0
        strip = max(3, int(min(h, w) * 0.045))
        border_density = np.mean(
            [
                edges[:strip, :].mean(),
                edges[-strip:, :].mean(),
                edges[:, :strip].mean(),
                edges[:, -strip:].mean(),
            ]
        )
        inner = edges[strip:-strip, strip:-strip]
        inner_density = float(inner.mean()) if inner.size else 0.0
        return float(np.clip((border_density - 0.55 * inner_density) / 0.10, 0.0, 1.0))

    def _apply_output_calibration(
        self,
        candidates: List[CandidateResult],
        image_shape: Tuple[int, int],
    ) -> List[CandidateResult]:
        """Optionally rerank final candidates for a known output style."""
        cfg = self.config.get("output_calibration", {})
        if not cfg.get("enabled", False) or not candidates:
            return candidates

        h, w = image_shape[:2]
        img_area = max(1, h * w)
        target_area = float(cfg.get("target_area_ratio", 0.25))
        top_n = max(1, int(cfg.get("top_n", 80)))
        score_weight = float(cfg.get("score_weight", 0.58))
        area_weight = float(cfg.get("area_weight", 0.32))
        aspect_weight = float(cfg.get("aspect_weight", 0.10))
        target_aspect = float(cfg.get("target_aspect_ratio", w / max(1, h)))

        ranked = list(candidates)
        pool = ranked[: min(top_n, len(ranked))]
        for cand in pool:
            x1, y1, x2, y2 = cand.bbox
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            area_ratio = (bw * bh) / img_area
            area_score = max(0.0, 1.0 - abs(area_ratio - target_area) / max(target_area, 1e-6))
            aspect = bw / max(1, bh)
            aspect_score = max(
                0.0,
                1.0 - abs(math.log(max(aspect, 1e-6) / max(target_aspect, 1e-6))) / math.log(3.0),
            )
            cand.final_score = float(
                score_weight * cand.final_score
                + area_weight * area_score
                + aspect_weight * aspect_score
            )

        ranked.sort(key=lambda c: c.final_score, reverse=True)
        return ranked

    def process_batch(
        self,
        image_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ) -> List[CropResult]:
        """Process all images in a directory.

        Args:
            image_dir: Directory containing input images.
            output_dir: Directory to save output files.
            extensions: Accepted image file extensions.

        Returns:
            List of CropResult for each processed image.
        """
        img_path = Path(image_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            f for f in img_path.iterdir() if f.suffix.lower() in extensions
        )

        results = []
        for img_file in image_files:
            result = self.process(str(img_file))
            results.append(result)

            # Save outputs
            name = img_file.stem
            # Visualization with bbox
            image = load_image(str(img_file))
            vis = draw_bbox(image, result.best_bbox, f"score={result.best_score:.3f}")
            save_image(vis, str(out_path / f"{name}_vis.jpg"))

            # Cropped image
            save_image(result.best_crop, str(out_path / f"{name}_crop.jpg"))

            # Coordinates file
            coord_file = out_path / f"{name}_coords.txt"
            coord_file.write_text(
                f"bbox: {result.best_bbox}\n"
                f"score: {result.best_score:.4f}\n"
                f"简短裁剪理由: {result.explanation}\n"
                f"完整分析报告: {result.explanation_full}\n",
                encoding="utf-8",
            )

        logger.info(f"Batch processed {len(results)} images -> {output_dir}")
        return results


    def process_with_custom_weights(
        self,
        image_path: str,
        custom_weights: Dict[str, float],
    ) -> CropResult:
        """Process an image with custom fusion weights (for ablation/grid search).

        Temporarily disables intent classification so that custom weights
        are not overridden by the strategy router.

        Args:
            image_path: Path to input image.
            custom_weights: Dict of dimension weights, e.g. {"aesthetic": 0.5, ...}.

        Returns:
            CropResult with custom-weighted fusion.
        """
        # Temporarily override fusion weights
        original_weights = {
            "aesthetic": self.fusion.weight_aesthetic,
            "saliency": self.fusion.weight_saliency,
            "composition": self.fusion.weight_composition,
            "subject": self.fusion.weight_subject,
            "technical": self.fusion.weight_technical,
            "area_prior": self.fusion.weight_area_prior,
            "roi_discard": self.fusion.weight_roi_discard,
            "semantic": self.fusion.weight_semantic,
            "subjectness": self.fusion.weight_subjectness,
            "artifact_avoidance": self.fusion.weight_artifact_avoidance,
        }

        for k, v in custom_weights.items():
            setattr(self.fusion, f"weight_{k}", v)

        try:
            return self.process(image_path)
        finally:
            for k, v in original_weights.items():
                setattr(self.fusion, f"weight_{k}", v)

    def visualize_result(self, image_path: str, result: CropResult) -> np.ndarray:
        """Create a visualization image with the best bbox + top-K overlays.

        Args:
            image_path: Path to the original image.
            result: CropResult from process().

        Returns:
            Visualization image (BGR).
        """
        image = load_image(image_path)
        # Draw top-K candidates in different colors
        top_bboxes = [c.bbox for c in result.top_candidates]
        labels = [
            f"#{i+1} score={c.final_score:.3f}"
            for i, c in enumerate(result.top_candidates)
        ]
        vis = draw_multiple_bboxes(image, top_bboxes, labels)
        return vis


def _clamped_min_size_bbox(bbox: BBox, h: int, w: int) -> BBox:
    x1, y1, x2, y2 = bbox
    bw, bh = max(8, x2 - x1), max(8, y2 - y1)
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h
    x1 = max(0, min(x1, max(0, w - bw)))
    y1 = max(0, min(y1, max(0, h - bh)))
    x2 = min(w, max(x1 + 8, x2))
    y2 = min(h, max(y1 + 8, y2))
    return (int(x1), int(y1), int(x2), int(y2))