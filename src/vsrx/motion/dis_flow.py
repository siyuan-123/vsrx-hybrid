from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vsrx.motion.global_registration import RegistrationResult
from vsrx.utils.geometry import clamp_bbox


@dataclass(slots=True)
class AlignedReference:
    image_bgr: np.ndarray
    source_mask: np.ndarray
    confidence: np.ndarray
    occlusion: np.ndarray
    flow_xy: np.ndarray
    roi_xyxy: tuple[int, int, int, int]
    registration: RegistrationResult


def _dis(preset: str) -> cv2.DISOpticalFlow:
    value = preset.lower()
    if "ultrafast" in value:
        mode = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    elif "medium" in value:
        mode = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    else:
        mode = cv2.DISOPTICAL_FLOW_PRESET_FAST
    engine = cv2.DISOpticalFlow_create(mode)
    engine.setUseSpatialPropagation(True)
    engine.setUseMeanNormalization(True)
    return engine


def _target_grid(roi: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    return xx, yy


def _global_warp_roi(
    source: np.ndarray,
    source_mask: np.ndarray,
    source_to_target: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = source.shape[:2]
    xx, yy = _target_grid(roi)
    try:
        inverse = np.linalg.inv(source_to_target.astype(np.float64))
    except np.linalg.LinAlgError:
        inverse = np.eye(3, dtype=np.float64)
    denominator = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
    denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
    map_x = ((inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]) / denominator).astype(
        np.float32
    )
    map_y = ((inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]) / denominator).astype(
        np.float32
    )
    valid = (map_x >= 0.0) & (map_y >= 0.0) & (map_x <= width - 1.001) & (map_y <= height - 1.001)
    warped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    warped_mask = cv2.remap(
        source_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped, warped_mask, valid


def _sample_flow(flow: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    channels = [
        cv2.remap(
            flow[..., channel], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        for channel in range(2)
    ]
    return np.stack(channels, axis=-1)


def align_reference_to_target(
    source: np.ndarray,
    target: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    registration: RegistrationResult,
    roi_xyxy: tuple[int, int, int, int],
    *,
    preset: str = "opencv_dis_fast",
    forward_backward_check: bool = True,
    fb_threshold_px: float = 1.8,
    use_local_flow: bool | None = None,
    local_flow_trigger_error: float = 0.105,
    local_flow_min_registration_score: float = 0.82,
) -> AlignedReference:
    """Align a source frame into target coordinates inside a bounded ROI.

    The returned residual flow is target->globally-warped-source, which is the
    direction required by ``cv2.remap``.
    """

    height, width = target.shape[:2]
    roi = clamp_bbox(roi_xyxy, width, height)
    x1, y1, x2, y2 = roi
    target_crop = target[y1:y2, x1:x2]
    target_mask_crop = target_mask[y1:y2, x1:x2]
    globally_warped, globally_warped_mask, global_valid = _global_warp_roi(
        source, source_mask, registration.transform, roi
    )

    target_gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(globally_warped, cv2.COLOR_BGR2GRAY)

    # Most subtitle videos need only global camera compensation.  Dense flow
    # is reserved for pairs whose globally aligned ring still disagrees.
    # This gate is the largest throughput win in the normal static/affine case.
    if use_local_flow is None:
        radius = max(3, int(round(min(target_crop.shape[:2]) * 0.035)))
        binary = (target_mask_crop > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        ring = (cv2.dilate(binary, kernel) > 0) & (binary == 0)
        allowed = ring & global_valid & (globally_warped_mask == 0)
        if int(allowed.sum()) < 96:
            allowed = global_valid & (globally_warped_mask == 0) & (target_mask_crop == 0)
        if int(allowed.sum()) >= 96:
            error = cv2.absdiff(target_gray, source_gray).astype(np.float32) / 255.0
            global_ring_error = float(np.median(error[allowed]))
        else:
            global_ring_error = 1.0
        use_local_flow = bool(
            registration.score < local_flow_min_registration_score
            or global_ring_error > local_flow_trigger_error
        )

    if min(target_crop.shape[:2]) < 12 or not use_local_flow:
        flow = np.zeros((*target_crop.shape[:2], 2), dtype=np.float32)
        local = globally_warped
        local_mask = globally_warped_mask
        fb_confidence = np.ones(target_crop.shape[:2], dtype=np.float32)
        local_valid = global_valid
    else:
        engine = _dis(preset)
        try:
            flow = engine.calc(target_gray, source_gray, None).astype(np.float32)
        except cv2.error:
            flow = np.zeros((*target_crop.shape[:2], 2), dtype=np.float32)
        yy, xx = np.mgrid[0 : target_crop.shape[0], 0 : target_crop.shape[1]].astype(np.float32)
        map_x = xx + flow[..., 0]
        map_y = yy + flow[..., 1]
        local_valid = (
            global_valid
            & (map_x >= 0.0)
            & (map_y >= 0.0)
            & (map_x <= target_crop.shape[1] - 1.001)
            & (map_y <= target_crop.shape[0] - 1.001)
        )
        local = cv2.remap(
            globally_warped, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
        )
        local_mask = cv2.remap(
            globally_warped_mask,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        if forward_backward_check:
            backward_engine = _dis(preset)
            try:
                backward = backward_engine.calc(source_gray, target_gray, None).astype(np.float32)
                sampled_backward = _sample_flow(backward, map_x, map_y)
                fb_error = np.linalg.norm(flow + sampled_backward, axis=-1)
                fb_confidence = np.exp(-np.square(fb_error / max(fb_threshold_px, 0.1))).astype(
                    np.float32
                )
                local_valid &= fb_error <= max(fb_threshold_px * 2.0, 2.0)
            except cv2.error:
                fb_confidence = np.full(target_crop.shape[:2], 0.65, dtype=np.float32)
        else:
            fb_confidence = np.ones(target_crop.shape[:2], dtype=np.float32)

    target_lab = cv2.cvtColor(target_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    local_lab = cv2.cvtColor(local, cv2.COLOR_BGR2LAB).astype(np.float32)
    photometric_error = np.mean(np.abs(target_lab - local_lab), axis=2) / 255.0
    photo_confidence = np.exp(-np.square(photometric_error / 0.16)).astype(np.float32)

    target_gray_f = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_gray_f = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gradient = cv2.Laplacian(target_gray_f, cv2.CV_32F)
    local_gradient = cv2.Laplacian(local_gray_f, cv2.CV_32F)
    gradient_error = np.abs(target_gradient - local_gradient) / (
        np.abs(target_gradient) + np.abs(local_gradient) + 12.0
    )
    gradient_confidence = np.exp(-np.square(gradient_error / 0.55)).astype(np.float32)

    source_unmasked = local_mask == 0
    # Photometric confidence is unreliable inside the target subtitle itself;
    # use a floor there and trust flow/registration/source visibility instead.
    photo_confidence = np.where(
        target_mask_crop > 0, np.maximum(photo_confidence, 0.62), photo_confidence
    )
    confidence = (
        fb_confidence * 0.48
        + photo_confidence * 0.28
        + gradient_confidence * 0.14
        + float(registration.score) * 0.10
    ).astype(np.float32)
    confidence *= local_valid.astype(np.float32)
    confidence *= source_unmasked.astype(np.float32)
    occlusion = ~(local_valid & source_unmasked & (confidence > 0.12))
    confidence[occlusion] = 0.0

    return AlignedReference(
        image_bgr=local,
        source_mask=local_mask,
        confidence=np.clip(confidence, 0.0, 1.0),
        occlusion=occlusion,
        flow_xy=flow,
        roi_xyxy=roi,
        registration=registration,
    )
