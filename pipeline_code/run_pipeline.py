from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

from .config import load_config
from .geo_utils import buffer_radii_px, make_circular_mask, pixel_area_m2
from .image_fetcher import ImageFetcher
from .model_inference import load_model
from .postprocess import decide, draw_overlay


def process_row(row: pd.Series, cfg, model_bundle) -> Dict[str, Any]:
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    sample_id = str(row.get("sample_id", row.name))

    fetcher = ImageFetcher(
        provider=cfg.imagery.provider,
        tile_size=cfg.imagery.tile_size,
        api_keys={
            "google": cfg.imagery.google_api_key,
            "mapbox": cfg.imagery.mapbox_api_key,
            "esri": cfg.imagery.esri_api_key,
        },
    )

    fetch = fetcher.fetch_image(lat, lon, cfg.imagery.zoom)
    image = fetch.image
    r1200_px, r2400_px = buffer_radii_px(lat, cfg.imagery.zoom)
    mask_1200 = make_circular_mask(image.shape[0], image.shape[1], r1200_px)
    mask_2400 = make_circular_mask(image.shape[0], image.shape[1], r2400_px)

    pv_probs, _ = model_bundle.predict_masks(image)
    pa_m2 = pixel_area_m2(lat, cfg.imagery.zoom)
    decision = decide(
        image=image,
        pv_probs=pv_probs,
        mask_1200=mask_1200,
        mask_2400=mask_2400,
        min_pv_pixels=cfg.thresholds.min_pv_pixels,
        default_confidence=cfg.thresholds.no_pv_confidence,
        pixel_area_m2=pa_m2,
        pv_prob_threshold=cfg.thresholds.pv_prob_thresh,
    )

    overlay = draw_overlay(image, decision, mask_1200, mask_2400)
    overlay_path = cfg.overlay_dir / f"{sample_id}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    record = {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": decision.has_solar,
        "pv_area_sqm_est": decision.pv_area_sqm_est,
        "buffer_radius_sqft": decision.buffer_radius_sqft,
        "confidence": decision.confidence,
        "qc_status": decision.qc_status,
        "polygons": [[list(pt) for pt in poly] for poly in decision.polygons],
        "image_metadata": fetch.metadata,
        "overlay_path": str(overlay_path),
    }
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EcoInnovators rooftop PV inference pipeline")
    parser.add_argument("--input_xlsx", required=True, help="Path to Excel with latitude/longitude columns")
    parser.add_argument("--output_dir", required=True, help="Directory to store predictions and overlays")
    parser.add_argument("--image_source", default="google", help="Imagery provider: google|mapbox")
    parser.add_argument("--zoom", type=int, default=20, help="Zoom level")
    parser.add_argument("--tile_size", type=int, default=640, help="Tile size in pixels")
    parser.add_argument("--model_path", default="trained_model/pv_segmentation.pt", help="Segmentation model path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        output_dir=args.output_dir,
        imagery_provider=args.image_source,
        zoom=args.zoom,
        tile_size=args.tile_size,
        model_path=args.model_path,
    )
    model_bundle = load_model(Path(args.model_path))

    df = pd.read_excel(args.input_xlsx)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="processing"):
        record = process_row(row, cfg, model_bundle)
        records.append(record)
        with cfg.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} records to {cfg.jsonl_path}")


if __name__ == "__main__":
    main()
