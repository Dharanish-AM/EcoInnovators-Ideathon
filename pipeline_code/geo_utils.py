import math
from typing import Tuple
import numpy as np

R_EARTH_M = 6378137.0
SQFT_TO_SQM = 0.092903


def meters_per_pixel(lat: float, zoom: int) -> float:
    lat_rad = math.radians(lat)
    return math.cos(lat_rad) * (2 * math.pi * R_EARTH_M) / (256 * (2**zoom))


def area_sqft_to_radius_m(area_sqft: float) -> float:
    area_sqm = area_sqft * SQFT_TO_SQM
    return math.sqrt(area_sqm / math.pi)


def buffer_radii_px(lat: float, zoom: int) -> Tuple[float, float]:
    mpp = meters_per_pixel(lat, zoom)
    r1200_m = area_sqft_to_radius_m(1200)
    r2400_m = area_sqft_to_radius_m(2400)
    return r1200_m / mpp, r2400_m / mpp


def make_circular_mask(height: int, width: int, radius_px: float) -> np.ndarray:
    cy = height // 2
    cx = width // 2
    y, x = np.ogrid[:height, :width]
    dist = (x - cx) ** 2 + (y - cy) ** 2
    return dist <= radius_px**2


def pixel_area_m2(lat: float, zoom: int) -> float:
    mpp = meters_per_pixel(lat, zoom)
    return mpp * mpp
