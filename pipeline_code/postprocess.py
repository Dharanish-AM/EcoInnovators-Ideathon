from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np


@dataclass
class Decision:
    has_solar: bool
    buffer_radius_sqft: Optional[int]
    confidence: float
    qc_status: str
    pv_area_sqm_est: float
    polygons: List[List[Tuple[int, int]]]


def choose_buffer(pv_mask: np.ndarray, mask_1200: np.ndarray, mask_2400: np.ndarray, min_pixels: int) -> tuple:
    pv_1200 = pv_mask & mask_1200
    pv_2400 = pv_mask & mask_2400
    count_1200 = int(pv_1200.sum())
    count_2400 = int(pv_2400.sum())
    if count_1200 >= min_pixels:
        return pv_1200, 1200, count_1200
    if count_2400 >= min_pixels:
        return pv_2400, 2400, count_2400
    return np.zeros_like(pv_mask, dtype=bool), None, 0


def mask_to_polygons(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[Tuple[int, int]]] = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon=1.0, closed=True)
        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if points:
            polygons.append(points)
    return polygons


def qc_status(image: np.ndarray, pv_pixel_count: int, min_pv_pixels: int) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    low_contrast = std < 8.0
    too_dark = mean < 25
    too_bright = mean > 230
    if pv_pixel_count < min_pv_pixels and (low_contrast or too_dark or too_bright):
        return "NOT_VERIFIABLE"
    return "VERIFIABLE"


def aggregate_confidence(pv_probs: np.ndarray, selected_mask: np.ndarray, default_no_pv: float) -> float:
    pv_pixels = pv_probs[selected_mask]
    if pv_pixels.size == 0:
        return default_no_pv
    return float(pv_pixels.mean())


def decide(
    image: np.ndarray,
    pv_probs: np.ndarray,
    mask_1200: np.ndarray,
    mask_2400: np.ndarray,
    min_pv_pixels: int,
    default_confidence: float,
    pixel_area_m2: float,
    pv_prob_threshold: float = 0.5,
) -> Decision:
    pv_mask = pv_probs > pv_prob_threshold
    selected_mask, buffer_sqft, pv_pixels = choose_buffer(pv_mask, mask_1200, mask_2400, min_pv_pixels)
    area = pv_pixels * pixel_area_m2
    polygons = mask_to_polygons(selected_mask)
    conf = aggregate_confidence(pv_probs, selected_mask, default_no_pv=default_confidence)
    qc = qc_status(image, pv_pixels, min_pv_pixels)
    return Decision(
        has_solar=buffer_sqft is not None,
        buffer_radius_sqft=buffer_sqft,
        confidence=conf,
        qc_status=qc,
        pv_area_sqm_est=area,
        polygons=polygons,
    )


def draw_overlay(
    image: np.ndarray,
    decision: Decision,
    mask_1200: np.ndarray,
    mask_2400: np.ndarray,
    color_pv: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    overlay = image.copy()
    circle_color = (255, 215, 0)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    if decision.buffer_radius_sqft == 1200:
        radius = int((mask_1200.astype("uint8") * 255).any(axis=0).sum() // 2)
    elif decision.buffer_radius_sqft == 2400:
        radius = int((mask_2400.astype("uint8") * 255).any(axis=0).sum() // 2)
    else:
        radius = 0
    if radius > 0:
        cv2.circle(overlay, center, radius, circle_color, 2)

    for poly in decision.polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=color_pv, thickness=2)
        cv2.fillPoly(overlay, [pts], color=(0, 128, 0))

    label = f"solar={decision.has_solar} area={decision.pv_area_sqm_est:.1f}sqm qc={decision.qc_status} conf={decision.confidence:.2f}"
    cv2.putText(overlay, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay
