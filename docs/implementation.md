# Implementation Notes

This repository packages an end-to-end pipeline for detecting rooftop solar PV around provided latitude/longitude coordinates.

## Flow
1. Read Excel input (`sample_id`, `latitude`, `longitude`).
2. Fetch satellite tile via configured provider.
3. Normalize to expected size; compute meters-per-pixel and buffer masks for 1200/2400 sq.ft.
4. Run PV segmentation model to get probability map.
5. Apply business rules: choose buffer with PV coverage above `min_pv_pixels`; compute area; set QC and confidence.
6. Export JSONL records and overlay images.

## Key modules
- `pipeline_code/run_pipeline.py`: CLI orchestration.
- `pipeline_code/image_fetcher.py`: imagery adapters (Google Static Maps, Mapbox).
- `pipeline_code/geo_utils.py`: geospatial conversions and masks.
- `pipeline_code/model_inference.py`: model loading and inference; includes dummy fallback.
- `pipeline_code/postprocess.py`: business rules, polygons, overlays, QC.
- `pipeline_code/config.py`: runtime configuration and defaults.

## Outputs
- JSONL at `<output_dir>/predictions.jsonl`.
- Overlays at `<output_dir>/overlays/`.
- Logs to stdout; extend as needed for MLflow/TensorBoard.

## Environment
- Python 3.10+.
- Dependencies listed in `environment_details/requirements.txt` or `environment_details/environment.yml`.

## Notes
- Imagery API keys must be provided via environment variables: `GOOGLE_MAPS_API_KEY`, `MAPBOX_API_KEY`, or `ESRI_API_KEY`.
- If no model weights exist, the dummy model yields empty masks so the pipeline still runs for structural validation.
