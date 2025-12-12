"""
Streamlit UI for EcoInnovators Rooftop Solar Detection Pipeline
Interactive interface to test and visualize the detection pipeline step-by-step.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image
import sys
import os
from dotenv import load_dotenv

# Load environment variables first
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Debug: Show what API key was loaded
mapbox_key = os.getenv("MAPBOX_API_KEY")
if not mapbox_key:
    st.error("⚠️ MAPBOX_API_KEY not found in environment. Trying to reload...")
    # Force reload
    from dotenv import dotenv_values
    env_vars = dotenv_values(env_path)
    for key, value in env_vars.items():
        os.environ[key] = value
    mapbox_key = os.getenv("MAPBOX_API_KEY")

# Add pipeline_code to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline_code"))

from pipeline_code.config import load_config
from pipeline_code.geo_utils import buffer_radii_px, make_circular_mask, pixel_area_m2
from pipeline_code.image_fetcher import ImageFetcher
from pipeline_code.model_inference import load_model
from pipeline_code.postprocess import decide, draw_overlay

st.set_page_config(
    page_title="EcoInnovators Solar Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☀️ Rooftop Solar PV Detection Pipeline")
st.markdown("Interactive testing and visualization tool")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Load real data from Excel
data_file = Path(__file__).parent / "data" / "EI_train_data.xlsx"

# Input Source Selection
input_mode = st.sidebar.radio("Input Source", ["Training Data", "Manual Input"])

if input_mode == "Training Data":
    if data_file.exists():
        df = pd.read_excel(data_file)
        st.sidebar.success(f"✅ Loaded {len(df)} real samples from training data")
        
        # Display sample info
        st.sidebar.markdown("**Real Training Samples:**")
        sample_idx = st.sidebar.selectbox(
            "Select Sample",
            range(len(df)),
            format_func=lambda i: f"{df.iloc[i].get('sample_id', f'Sample {i}')} ({df.iloc[i].get('latitude', 0):.4f}, {df.iloc[i].get('longitude', 0):.4f})"
        )
        row = df.iloc[sample_idx]
        sample_id = str(row.get("sample_id", f"sample_{sample_idx}"))
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    else:
        st.sidebar.error("❌ data/EI_train_data.xlsx not found!")
        st.sidebar.info("Please upload your training data Excel file to the data folder.")
        uploaded_file = st.sidebar.file_uploader("Or upload Excel file", type=["xlsx", "xls"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Loaded {len(df)} samples")
            sample_idx = st.sidebar.selectbox("Select Sample", range(len(df)))
            row = df.iloc[sample_idx]
            sample_id = str(row.get("sample_id", f"sample_{sample_idx}"))
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        else:
            lat, lon, sample_id = None, None, None

else:  # Manual Input
    st.sidebar.markdown("**Manual Coordinates:**")
    lat_input = st.sidebar.number_input("Latitude", value=37.7749, format="%.6f")
    lon_input = st.sidebar.number_input("Longitude", value=-122.4194, format="%.6f")
    
    confirm_manual = st.sidebar.checkbox("Confirm Coordinates")
    
    if confirm_manual:
        lat = lat_input
        lon = lon_input
        sample_id = "manual_input"
    else:
        lat, lon, sample_id = None, None, None

# Provider settings
imagery_provider = st.sidebar.selectbox("Imagery Provider", ["mapbox", "google"])
zoom = st.sidebar.slider("Zoom Level", 18, 21, 20)
tile_size = st.sidebar.selectbox("Tile Size", [640, 512, 768], index=0)

# Model settings
model_path = st.sidebar.text_input("Model Path", "trained_model/pv_segmentation.pt")

# Threshold settings
st.sidebar.subheader("Detection Thresholds")
min_pv_pixels = st.sidebar.slider("Min PV Pixels", 10, 200, 40)
pv_prob_thresh = st.sidebar.slider("PV Probability Threshold", 0.0, 1.0, 0.5, 0.05)

# Main processing
if lat is not None and lon is not None:
    if st.sidebar.button("🚀 Run Detection Pipeline", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Step 1: Configuration
                st.header("📋 Step 1: Configuration")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sample ID", sample_id)
                    st.metric("Latitude", f"{lat:.6f}")
                with col2:
                    st.metric("Longitude", f"{lon:.6f}")
                    st.metric("Zoom Level", zoom)
                with col3:
                    st.metric("Provider", imagery_provider)
                    st.metric("Tile Size", f"{tile_size}×{tile_size}")
                
                # Step 2: Fetch imagery
                st.header("🛰️ Step 2: Fetch Satellite Imagery")
                
                # Ensure API keys are loaded
                mapbox_api = os.getenv("MAPBOX_API_KEY")
                google_api = os.getenv("GOOGLE_MAPS_API_KEY")
                
                fetcher = ImageFetcher(
                    provider=imagery_provider,
                    tile_size=tile_size,
                    api_keys={
                        "mapbox": mapbox_api,
                        "google": google_api,
                        "esri": os.getenv("ESRI_API_KEY"),
                    },
                )
                
                fetch_result = fetcher.fetch_image(lat, lon, zoom)
                image = fetch_result.image
                
                # Debug output
                st.write(f"**Debug:** Image shape: {image.shape}, Mean pixel value: {image.mean():.1f}")
                
                # Check if we got a valid image
                if fetch_result.metadata.get("source") == "blank":
                    st.error(f"Failed to fetch image: {fetch_result.metadata.get('reason', 'Unknown error')}")
                    if imagery_provider == "mapbox" and not mapbox_api:
                        st.error("🔑 Mapbox API key is missing! Please add MAPBOX_API_KEY to .env")
                    elif imagery_provider == "google" and not google_api:
                        st.error("🔑 Google Maps API key is missing! Please add GOOGLE_MAPS_API_KEY to .env")
                    st.info("Make sure your API key is configured in the `.env` file")
                    st.stop()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Satellite Tile", width='stretch')
                with col2:
                    st.json(fetch_result.metadata)
                
                # Step 3: Geospatial calculations
                st.header("🌍 Step 3: Geospatial Calculations")
                
                r1200_px, r2400_px = buffer_radii_px(lat, zoom)
                mask_1200 = make_circular_mask(image.shape[0], image.shape[1], r1200_px)
                mask_2400 = make_circular_mask(image.shape[0], image.shape[1], r2400_px)
                pa_m2 = pixel_area_m2(lat, zoom)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("1200 sq.ft radius", f"{r1200_px:.1f} px")
                    st.metric("2400 sq.ft radius", f"{r2400_px:.1f} px")
                with col2:
                    st.metric("Pixel area", f"{pa_m2:.6f} m²")
                    st.metric("1200 buffer area", f"{mask_1200.sum() * pa_m2:.1f} m²")
                with col3:
                    st.metric("2400 buffer area", f"{mask_2400.sum() * pa_m2:.1f} m²")
                
                # Visualize masks
                mask_viz = np.zeros_like(image)
                mask_viz[mask_1200] = [255, 215, 0]  # Gold for 1200
                mask_viz[mask_2400 & ~mask_1200] = [100, 100, 255]  # Blue for 2400
                overlay_masks = (image * 0.6 + mask_viz * 0.4).astype(np.uint8)
                
                st.image(overlay_masks, caption="Buffer Zones (Gold: 1200 sq.ft, Blue: 2400 sq.ft)", width='stretch')
                
                # Step 4: Model inference
                st.header("🤖 Step 4: ML Model Inference")
                
                model_bundle = load_model(Path(model_path))
                pv_probs, _ = model_bundle.predict_masks(image)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(pv_probs, caption="PV Probability Map", width='stretch', clamp=True)
                    st.metric("Max Probability", f"{pv_probs.max():.3f}")
                with col2:
                    pv_binary = (pv_probs > pv_prob_thresh).astype(np.uint8) * 255
                    st.image(pv_binary, caption=f"Binary Mask (threshold={pv_prob_thresh})", width='stretch')
                    st.metric("PV Pixels Detected", int((pv_probs > pv_prob_thresh).sum()))
                
                # Debug info
                st.info(f"📊 **Debug Info**\n"
                        f"- Probability threshold used: {pv_prob_thresh}\n"
                        f"- Pixels above threshold: {(pv_probs > pv_prob_thresh).sum()}\n"
                        f"- Min PV pixels required: {min_pv_pixels}\n"
                        f"- Mean probability in image: {pv_probs.mean():.3f}\n"
                        f"- Std dev: {pv_probs.std():.3f}")
                
                # Step 5: Decision logic
                st.header("🎯 Step 5: Decision Logic")
                
                decision = decide(
                    image=image,
                    pv_probs=pv_probs,
                    mask_1200=mask_1200,
                    mask_2400=mask_2400,
                    min_pv_pixels=min_pv_pixels,
                    default_confidence=0.1,
                    pixel_area_m2=pa_m2,
                    pv_prob_threshold=pv_prob_thresh,
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Solar Detected", "✅ Yes" if decision.has_solar else "❌ No")
                with col2:
                    st.metric("Buffer Used", f"{decision.buffer_radius_sqft} sq.ft" if decision.buffer_radius_sqft else "N/A")
                with col3:
                    st.metric("PV Area", f"{decision.pv_area_sqm_est:.2f} m²")
                with col4:
                    st.metric("Confidence", f"{decision.confidence:.2%}")
                
                st.metric("QC Status", decision.qc_status)
                
                if decision.polygons:
                    st.success(f"Detected {len(decision.polygons)} PV polygon(s)")
                    with st.expander("View Polygon Coordinates"):
                        for i, poly in enumerate(decision.polygons):
                            st.write(f"Polygon {i+1}: {len(poly)} vertices")
                            st.json(poly[:5])  # Show first 5 points
                
                # Step 6: Final overlay
                st.header("🖼️ Step 6: Final Visualization")
                
                overlay = draw_overlay(image, decision, mask_1200, mask_2400)
                st.image(overlay, caption="Final Detection Overlay", width='stretch')
                
                # Step 7: JSON output
                st.header("📄 Step 7: JSON Output")
                
                output_record = {
                    "sample_id": sample_id,
                    "lat": lat,
                    "lon": lon,
                    "has_solar": decision.has_solar,
                    "pv_area_sqm_est": decision.pv_area_sqm_est,
                    "buffer_radius_sqft": decision.buffer_radius_sqft,
                    "confidence": decision.confidence,
                    "qc_status": decision.qc_status,
                    "polygons": [[list(pt) for pt in poly] for poly in decision.polygons],
                    "image_metadata": fetch_result.metadata,
                }
                
                st.json(output_record)
                
                # Download button
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(output_record, indent=2),
                    file_name=f"{sample_id}_detection.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.exception(e)
else:
    st.info("👈 Configure input coordinates in the sidebar and click 'Run Detection Pipeline'")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**EcoInnovators Solar Detection**

This tool demonstrates the complete pipeline:
1. Configuration loading
2. Satellite imagery fetching
3. Geospatial calculations
4. ML model inference
5. Business logic decisions
6. Visualization & output

Built for the EcoInnovators Ideathon.
""")
