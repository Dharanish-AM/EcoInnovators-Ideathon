from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ThresholdConfig:
    pv_prob_thresh: float = 0.5
    min_pv_pixels: int = 40
    min_roof_pixels: int = 100
    no_pv_confidence: float = 0.1


@dataclass
class ImageryConfig:
    provider: str = "google"
    zoom: int = 20
    tile_size: int = 640
    google_api_key: Optional[str] = None
    mapbox_api_key: Optional[str] = None
    esri_api_key: Optional[str] = None


@dataclass
class RunConfig:
    thresholds: ThresholdConfig
    imagery: ImageryConfig
    model_path: Path
    output_dir: Path
    overlay_dir: Path
    jsonl_path: Path


def load_config(
    output_dir: str,
    imagery_provider: str = "google",
    zoom: int = 20,
    tile_size: int = 640,
    model_path: str | Path = "trained_model/pv_segmentation.pt",
) -> RunConfig:
    output = Path(output_dir)
    overlay_dir = output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "predictions.jsonl"

    imagery = ImageryConfig(
        provider=imagery_provider,
        zoom=zoom,
        tile_size=tile_size,
        google_api_key=os.getenv("GOOGLE_MAPS_API_KEY"),
        mapbox_api_key=os.getenv("MAPBOX_API_KEY"),
        esri_api_key=os.getenv("ESRI_API_KEY"),
    )

    thresholds = ThresholdConfig()

    return RunConfig(
        thresholds=thresholds,
        imagery=imagery,
        model_path=Path(model_path),
        output_dir=output,
        overlay_dir=overlay_dir,
        jsonl_path=jsonl_path,
    )
