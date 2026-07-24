from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

from vsrx.utils.config import Config


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """A transform mapping source-image coordinates into target coordinates."""

    transform: np.ndarray
    method: str
    score: float
    valid_fraction: float


class GlobalRegistrar:
    """Fast, robust source-to-target image registration.

    The normal path is intentionally cheap: identity and phase-correlation
    translation are evaluated first and accepted immediately when they explain
    the frame pair well.  ECC and ORB are only attempted for pairs that do not
    pass the fast gate.  Every candidate is evaluated in one explicit
    source->target convention, avoiding OpenCV ECC inverse-warp ambiguity.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._cache: dict[tuple[int, int, int], RegistrationResult] = {}
        self._cache_lock = Lock()

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return gray.astype(np.uint8, copy=False)

    def _resize_pair(
        self,
        source: np.ndarray,
        target: np.ndarray,
        exclude_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
        maximum = int(
            self.config.get("motion_analysis.global_registration.downscale_max_side", 960)
        )
        height, width = target.shape[:2]
        scale = min(1.0, maximum / max(height, width)) if maximum > 0 else 1.0
        if scale >= 0.999:
            return self._gray(source), self._gray(target), exclude_mask, 1.0
        size = (max(32, int(round(width * scale))), max(32, int(round(height * scale))))
        source_small = cv2.resize(self._gray(source), size, interpolation=cv2.INTER_AREA)
        target_small = cv2.resize(self._gray(target), size, interpolation=cv2.INTER_AREA)
        mask_small = None
        if exclude_mask is not None:
            mask_small = cv2.resize(
                (exclude_mask > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            )
        return source_small, target_small, mask_small, scale

    @staticmethod
    def _small_to_full(transform: np.ndarray, scale: float) -> np.ndarray:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        if scale == 1.0:
            return matrix
        scaling = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        return np.linalg.inv(scaling) @ matrix @ scaling

    @staticmethod
    def _warp(
        source: np.ndarray, transform: np.ndarray, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = shape
        warped = cv2.warpPerspective(
            source,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpPerspective(
            np.ones(source.shape[:2], dtype=np.uint8) * 255,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped, valid

    @classmethod
    def _evaluate(
        cls,
        source: np.ndarray,
        target: np.ndarray,
        transform: np.ndarray,
        exclude_mask: np.ndarray | None,
    ) -> tuple[float, float]:
        warped, valid = cls._warp(source, transform, target.shape[:2])
        allowed = valid > 0
        if exclude_mask is not None:
            allowed &= exclude_mask == 0
        # Overlap is a geometric quantity; do not let a dark scene look like
        # a low-overlap transform.  Dark-dark pixels are only omitted from the
        # photometric statistic below.
        valid_fraction = float(allowed.mean())
        evaluation = allowed & ((target > 3) | (warped > 3))
        if int(evaluation.sum()) >= 128:
            allowed = evaluation
        if allowed.sum() < 128:
            return 1e6, valid_fraction
        delta = cv2.absdiff(warped, target).astype(np.float32) / 255.0
        photo_values = delta[allowed]
        # Median alone cannot distinguish translations in large flat regions;
        # combine robust location, upper quartile and bounded mean.
        photometric = float(
            np.median(photo_values) * 0.45
            + np.quantile(photo_values, 0.75) * 0.35
            + np.mean(photo_values) * 0.20
        )
        source_grad = cv2.Laplacian(warped, cv2.CV_32F)
        target_grad = cv2.Laplacian(target, cv2.CV_32F)
        grad_delta = np.abs(source_grad - target_grad)
        grad_scale = np.abs(target_grad) + np.abs(source_grad) + 4.0
        grad_values = (grad_delta / grad_scale)[allowed]
        gradient = float(np.median(grad_values) * 0.45 + np.mean(grad_values) * 0.55)
        overlap_penalty = max(0.0, 0.55 - valid_fraction) * 2.0
        return photometric * 0.72 + gradient * 0.28 + overlap_penalty, valid_fraction

    @staticmethod
    def _phase_preprocess(image: np.ndarray, excluded: np.ndarray | None) -> np.ndarray:
        value = image.astype(np.float32) / 255.0
        # Gradients are much less sensitive to exposure changes than intensity.
        gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
        value = cv2.magnitude(gx, gy)
        if excluded is not None:
            valid = excluded == 0
            if int(valid.sum()) >= 128:
                fill = float(np.median(value[valid]))
                value = value.copy()
                value[~valid] = fill
        value -= float(value.mean())
        std = float(value.std())
        if std > 1e-6:
            value /= std
        # Hanning suppresses wrap-around peaks at frame borders.
        window = cv2.createHanningWindow((value.shape[1], value.shape[0]), cv2.CV_32F)
        return np.ascontiguousarray(value * window, dtype=np.float32)

    def _phase_candidates(
        self,
        source: np.ndarray,
        target: np.ndarray,
        exclude_mask: np.ndarray | None,
    ) -> list[np.ndarray]:
        try:
            source_phase = self._phase_preprocess(source, exclude_mask)
            target_phase = self._phase_preprocess(target, exclude_mask)
            (dx, dy), response = cv2.phaseCorrelate(source_phase, target_phase)
        except (cv2.error, ValueError):
            return []
        if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(response):
            return []
        maximum_shift = float(
            self.config.get(
                "motion_analysis.global_registration.phase_max_shift_ratio",
                0.35,
            )
        ) * max(source.shape)
        if abs(dx) > maximum_shift or abs(dy) > maximum_shift:
            return []
        forward = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        # OpenCV's documented convention is source->target for phaseCorrelate,
        # but evaluating both directions costs almost nothing and protects us
        # from backend/version convention mistakes.
        inverse = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return [forward, inverse]

    def _ecc_candidate(
        self,
        source: np.ndarray,
        target: np.ndarray,
        exclude_mask: np.ndarray | None,
    ) -> list[np.ndarray]:
        iterations = int(self.config.get("motion_analysis.global_registration.ecc_iterations", 50))
        epsilon = float(self.config.get("motion_analysis.global_registration.ecc_epsilon", 1e-4))
        warp = np.eye(2, 3, dtype=np.float32)
        input_mask = None if exclude_mask is None else (exclude_mask == 0).astype(np.uint8) * 255
        try:
            cv2.findTransformECC(
                target.astype(np.float32) / 255.0,
                source.astype(np.float32) / 255.0,
                warp,
                cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon),
                inputMask=input_mask,
                gaussFiltSize=5,
            )
        except cv2.error:
            return []
        matrix = np.vstack([warp.astype(np.float64), [0.0, 0.0, 1.0]])
        candidates = [matrix]
        with suppress(np.linalg.LinAlgError):
            candidates.append(np.linalg.inv(matrix))
        return candidates

    @staticmethod
    def _orb_candidate(
        source: np.ndarray, target: np.ndarray, exclude_mask: np.ndarray | None
    ) -> np.ndarray | None:
        feature_mask = None if exclude_mask is None else (exclude_mask == 0).astype(np.uint8) * 255
        orb = cv2.ORB_create(nfeatures=1400, scaleFactor=1.2, nlevels=8, fastThreshold=12)
        src_keypoints, src_desc = orb.detectAndCompute(source, feature_mask)
        dst_keypoints, dst_desc = orb.detectAndCompute(target, feature_mask)
        if src_desc is None or dst_desc is None or len(src_keypoints) < 8 or len(dst_keypoints) < 8:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(src_desc, dst_desc, k=2)
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]
        if len(good) < 8:
            return None
        source_points = np.float32([src_keypoints[item.queryIdx].pt for item in good])
        target_points = np.float32([dst_keypoints[item.trainIdx].pt for item in good])
        affine, inliers = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        if affine is None or inliers is None or int(inliers.sum()) < 6:
            return None
        return np.vstack([affine.astype(np.float64), [0.0, 0.0, 1.0]])

    @staticmethod
    def _mask_signature(mask: np.ndarray | None) -> int:
        if mask is None or mask.size == 0:
            return 0
        # Cheap signature sufficient to invalidate expanded-mask retries.
        sample = mask[:: max(1, mask.shape[0] // 32), :: max(1, mask.shape[1] // 32)]
        return int((int(np.count_nonzero(mask)) * 1315423911 + int(sample.sum())) & 0x7FFFFFFF)

    def _result_from_best(
        self,
        best_method: str,
        best_matrix_small: np.ndarray,
        best_error: float,
        best_valid: float,
        scale: float,
    ) -> RegistrationResult:
        full_matrix = self._small_to_full(best_matrix_small, scale).astype(np.float32)
        score = float(np.clip(1.0 - best_error / 0.35, 0.0, 1.0))
        return RegistrationResult(full_matrix, best_method, score, best_valid)

    def estimate(
        self,
        source: np.ndarray,
        target: np.ndarray,
        *,
        source_key: int | None = None,
        target_key: int | None = None,
        exclude_mask: np.ndarray | None = None,
    ) -> RegistrationResult:
        mask_signature = self._mask_signature(exclude_mask)
        cache_key = (
            None
            if source_key is None or target_key is None
            else (int(source_key), int(target_key), mask_signature)
        )
        if cache_key is not None:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        source_small, target_small, mask_small, scale = self._resize_pair(
            source, target, exclude_mask
        )
        methods = list(
            self.config.get(
                "motion_analysis.global_registration.method_order",
                ["phase_translation", "ecc_affine", "orb_ransac_affine", "identity"],
            )
        )
        fast_error = float(
            self.config.get("motion_analysis.global_registration.fast_accept_error", 0.095)
        )
        fast_valid = float(
            self.config.get("motion_analysis.global_registration.fast_accept_valid_fraction", 0.70)
        )
        expensive_error = float(
            self.config.get("motion_analysis.global_registration.expensive_accept_error", 0.075)
        )

        best_method = "identity"
        best_matrix = np.eye(3, dtype=np.float64)
        best_error, best_valid = self._evaluate(source_small, target_small, best_matrix, mask_small)

        def consider(method: str, matrix: np.ndarray) -> None:
            nonlocal best_method, best_matrix, best_error, best_valid
            error, valid = self._evaluate(source_small, target_small, matrix, mask_small)
            if error < best_error:
                best_method, best_matrix, best_error, best_valid = method, matrix, error, valid

        # Phase correlation is the highest-value fast path.  Identity has
        # already been evaluated, so static frames also terminate here.
        if "phase_translation" in methods:
            for matrix in self._phase_candidates(source_small, target_small, mask_small):
                consider("phase_translation", matrix)
        if best_error <= fast_error and best_valid >= fast_valid:
            result = self._result_from_best(best_method, best_matrix, best_error, best_valid, scale)
            if cache_key is not None:
                with self._cache_lock:
                    self._cache[cache_key] = result
            return result

        # Only difficult frame pairs pay for ECC/ORB.
        for method in methods:
            if method in {"phase_translation", "identity"}:
                continue
            if method == "ecc_affine":
                for matrix in self._ecc_candidate(source_small, target_small, mask_small):
                    consider(method, matrix)
            elif method == "orb_ransac_affine":
                matrix = self._orb_candidate(source_small, target_small, mask_small)
                if matrix is not None:
                    consider(method, matrix)
            if best_error <= expensive_error and best_valid >= fast_valid:
                break

        result = self._result_from_best(best_method, best_matrix, best_error, best_valid, scale)
        if cache_key is not None:
            with self._cache_lock:
                self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
