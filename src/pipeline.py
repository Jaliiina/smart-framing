"""Top-level AestheticCropper pipeline orchestrator."""

from __future__ import annotations

import logging
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

    def process(self, image_path: str) -> CropResult:
        """Process a single image through the full pipeline.

        Args:
            image_path: Path to input image.

        Returns:
            CropResult with best bbox, crop, scores, explanation, etc.
        """
        start_time = time.time()
        image = load_image(image_path)
        h, w = image.shape[:2]

        # --- Step 1: Run U2-Net once → saliency map ---
        saliency_map, is_uniform = self.saliency_det.detect(image)

        # --- Step 2: Run YOLOv8 once → detected objects ---
        detected_objects = self.subject_det.detect(image)
        has_subject = len(detected_objects) > 0

        # --- Step 3: Generate candidates (grid + saliency-guided) ---
        candidates = self.candidate_gen.generate(image, saliency_map)
        logger.info(f"Generated {len(candidates)} candidates for {image_path}")

        if len(candidates) == 0:
            # Fallback: use the whole image
            candidates = [(0, 0, w, h)]

        # --- Step 4: Score each candidate ---
        # 4a. Aesthetic scores
        aesthetic_scores = self.aesthetic_scorer.score_candidates(image, candidates)

        # 4b. Saliency preservation scores
        saliency_scores = self.saliency_det.score_candidates(
            saliency_map, candidates, image.shape
        )

        # 4c. Composition scores
        composition_scores = self.comp_scorer.score_candidates(
            image, candidates, saliency_map, detected_objects
        )

        # 4d. Subject completeness scores
        subject_scores = self.subject_det.score_candidates(
            candidates, detected_objects, image.shape
        )

        # 4e. Technical quality scores
        technical_scores = self.tech_scorer.score_candidates(image, candidates)

        # --- Step 5: Fuse scores and select best ---
        best, top_k = self.fusion.fuse(
            bboxes=candidates,
            aesthetic_scores=aesthetic_scores,
            saliency_scores=saliency_scores,
            composition_scores=composition_scores,
            subject_scores=subject_scores,
            technical_scores=technical_scores,
            saliency_is_uniform=is_uniform,
            has_subject=has_subject,
        )

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
        }

        for k, v in custom_weights.items():
            setattr(self.fusion, f"weight_{k}", v)

        try:
            return self.process(image_path)
        finally:
            # Restore original weights
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
