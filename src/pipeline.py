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

        self.candidate_gen = CandidateGenerator(self.config)
        self.saliency_det = SaliencyDetector(self.config)
        self.aesthetic_scorer = AestheticScorer(self.config)
        self.subject_det = SubjectDetector(self.config)
        self.comp_scorer = CompositionScorer(self.config)
        self.tech_scorer = TechnicalQualityScorer(self.config)
        self.fusion = FusionModule(self.config)
        self.explainer = ExplanationGenerator(self.config)

        # Print model info
        u2net_cfg = self.config.get("models", {}).get("u2net", {})
        yolo_cfg = self.config.get("models", {}).get("yolo", {})
        u2net_path = u2net_cfg.get("weights_path" if not u2net_cfg.get("use_lite", False) else "lite_weights_path", "models/u2netp.pth")
        u2net_name = "U2Net" if not u2net_cfg.get("use_lite", False) else "U2NetP (lite)"
        yolo_model = yolo_cfg.get("model_name", "yolov8n.pt")
        yolo_conf = yolo_cfg.get("confidence_threshold", 0.3)
        aesthetic_device = self.config.get("aesthetic", {}).get("device", "cuda")
        u2net_device = u2net_cfg.get("device", "cuda")
        yolo_device = yolo_cfg.get("device", "cuda")
        print(
            f"[Model Info] saliency={u2net_name} [{u2net_path}]; "
            f"subject={yolo_model} (conf={yolo_conf}); "
            f"aesthetic=CLIP zero-shot prompts (ViT-L/14) [{self.aesthetic_scorer.model_path if hasattr(self.aesthetic_scorer, 'model_path') else 'models/aesthetic_predictor.pth'}]; "
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

        # --- Step 1: Run dual saliency (U2-Net + fallback) ---
        saliency_map, fallback_sal_map, is_uniform, fallback_uniform = (
            self.saliency_det.detect_dual(image)
        )

        # --- Step 2: Run YOLOv8 once → detected objects ---
        detected_objects = self.subject_det.detect(image)
        has_subject = len(detected_objects) > 0

        # --- Step 3: Generate candidates (grid + saliency-guided) ---
        candidates = self.candidate_gen.generate(image, saliency_map)
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

        # 4e. Technical quality scores
        technical_scores = self.tech_scorer.score_candidates(image, candidates)

        # --- Step 5: Fuse scores and select best ---
        best, top_k, all_ranked = self.fusion.fuse(
            bboxes=candidates,
            aesthetic_scores=aesthetic_scores,
            saliency_scores=saliency_scores,
            composition_scores=composition_scores,
            subject_scores=subject_scores,
            technical_scores=technical_scores,
            saliency_is_uniform=is_uniform,
            has_subject=has_subject,
            image_shape=image.shape[:2],
            saliency_map=saliency_map,
            return_all=True,  # Get all ranked candidates for evaluation
            dual_saliency_scores=dual_saliency_scores,
        )
        
        # Use all_ranked if available, otherwise fall back to candidates
        all_candidates = all_ranked if all_ranked else candidates

        # --- Step 6: Generate explanation ---
        explanation = self.explainer.generate(best.sub_scores, has_subject)

        # --- Step 7: Build result ---
        x1, y1, x2, y2 = best.bbox
        best_crop = image[y1:y2, x1:x2]

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
            explanation=explanation,
            saliency_map=saliency_map,
            detected_objects=detected_objects,
            all_candidates=all_candidates,
        )

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
                f"explanation: {result.explanation}\n",
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
