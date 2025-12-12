from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import requests
from PIL import Image
import numpy as np


@dataclass
class FetchResult:
    image: np.ndarray
    metadata: Dict[str, Any]


class ImageFetcher:
    def __init__(self, provider: str, tile_size: int, api_keys: Dict[str, str | None]):
        self.provider = provider
        self.tile_size = tile_size
        self.api_keys = api_keys

    def fetch_image(self, lat: float, lon: float, zoom: int) -> FetchResult:
        if self.provider == "google":
            return self._fetch_google(lat, lon, zoom)
        if self.provider == "mapbox":
            return self._fetch_mapbox(lat, lon, zoom)
        return self._blank_tile(lat, lon, zoom, reason="unknown_provider")

    def _fetch_google(self, lat: float, lon: float, zoom: int) -> FetchResult:
        key = self.api_keys.get("google")
        if not key:
            return self._blank_tile(lat, lon, zoom, reason="missing_google_api_key")
        url = (
            "https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={zoom}&size={self.tile_size}x{self.tile_size}&maptype=satellite&key={key}"
        )
        return self._download(url, lat, lon, zoom, source="google_static_maps")

    def _fetch_mapbox(self, lat: float, lon: float, zoom: int) -> FetchResult:
        key = self.api_keys.get("mapbox")
        if not key:
            return self._blank_tile(lat, lon, zoom, reason="missing_mapbox_api_key")
        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
            f"{lon},{lat},{zoom}/{self.tile_size}x{self.tile_size}?access_token={key}"
        )
        return self._download(url, lat, lon, zoom, source="mapbox_static")

    def _download(self, url: str, lat: float, lon: float, zoom: int, source: str) -> FetchResult:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.array(img)
            # Validate we got a real image (not all black or white)
            mean_pixel = arr.mean()
            if mean_pixel < 5:  # Image is too dark (likely error response)
                return self._blank_tile(lat, lon, zoom, reason=f"Image too dark (mean={mean_pixel:.1f}), possible API error")
            return FetchResult(image=arr, metadata={"source": source, "zoom": zoom, "url": url})
        except Exception as exc:  # noqa: BLE001
            return self._blank_tile(lat, lon, zoom, reason=str(exc))

    def _blank_tile(self, lat: float, lon: float, zoom: int, reason: str) -> FetchResult:
        arr = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        return FetchResult(
            image=arr,
            metadata={"source": "blank", "zoom": zoom, "reason": reason, "lat": lat, "lon": lon},
        )
