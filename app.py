# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# 31 FEATURES: 6 Meristic + 4 Morphometric + 21 Truss Individual
# WITH SPECIES IMAGES
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# SIDEBAR - MODE SELECTION
# ===============================

st.sidebar.title("🐟 Mugilidae Fish Classifier")
st.sidebar.markdown("---")

# Mode selection
data_mode = st.sidebar.radio(
    "📊 Data Mode",
    ["⚖️ Balanced Data (200 per species)", "🔬 Real Data Only (Original)"],
    help="Balanced: 200 specimens per species (recommended)\nReal Only: Original imbalanced data (9-84 specimens)"
)

st.sidebar.markdown("---")

st.sidebar.header("📋 About")
st.sidebar.info("""
Comparative Study Results (31 Features):
- 🥇 ANN-GWO: Best accuracy
- 🥈 ANN: 
- 🥉 ANN-PSO: 
- ANN-GA: 

31 Features: Meristic (6), Morphometric (4), Truss (21)

Best Architecture: ANN-GWO (optimized)
""")

st.sidebar.markdown("---")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### 31 Features: 6 Meristic + 4 Morphometric + 21 Truss Individual")
st.markdown("---")

# Show which mode is active
if data_mode == "⚖️ Balanced Data (200 per species)":
    st.info("📌 Active Mode: Balanced Dataset (200 specimens per species) - Higher accuracy")
else:
    st.warning("📌 Active Mode: Real Data Only (Original imbalanced data: 9-84 specimens per species) - Lower accuracy")

# ===============================
# LOAD MODELS (31 FEATURES)
# ===============================

@st.cache_resource
def load_all_models():
    """Load all trained models from .pkl files (31 features)"""
    models = {}
    try:
        models['ann'] = joblib.load('ann_model_31features.pkl')
        models['pso'] = joblib.load('pso_model_31features.pkl')
        models['ga'] = joblib.load('ga_model_31features.pkl')
        models['gwo'] = joblib.load('gwo_model_31features.pkl')
        models['scaler'] = joblib.load('scaler_31features.pkl')
        models['label_encoder'] = joblib.load('label_encoder_31features.pkl')
        models['feature_names'] = joblib.load('feature_names_31features.pkl')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all .pkl files are uploaded to GitHub")
        return None

models = load_all_models()

if models is not None:
    
    FEATURE_NAMES = models['feature_names']
    label_encoder = models['label_encoder']
    scaler = models['scaler']
    species_names = label_encoder.classes_
    
    st.success("✅ 31-feature models loaded successfully!")
    
    # ===============================
    # MODEL PERFORMANCE TABLE
    # ===============================
    
    st.header("📊 Model Performance Comparison (31 Features)")
    
    # Temporary placeholder - replace with actual results after training
    results_data = {
        'Method': ['ANN', 'ANN-PSO', 'ANN-GA', 'ANN-GWO 🏆'],
        'Architecture': ['(20,10)', 'Optimized', 'Optimized', 'Optimized'],
        'Test Accuracy': ['Loading...', 'Loading...', 'Loading...', 'Loading...'],
        'Accuracy': [0.0, 0.0, 0.0, 0.0],
        'Training Time': ['~10 min', '~25 min', '~25 min', '~25 min']
    }
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)
    
    # ===============================
    # PREDICTION SECTION - 31 FEATURES
    # ===============================
    
    st.header("🔮 Identify Fish Species (31 Features)")
    
    if data_mode == "⚖️ Balanced Data (200 per species)":
        st.info(f"🎯 Using Balanced Data Mode (200 specimens per species)")
    else:
        st.info(f"🎯 Using Real Data Only Mode")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model for Prediction",
        options=[
            "ANN-GWO 🏆 (Recommended - Best)",
            "ANN",
            "ANN-PSO", 
            "ANN-GA"
        ],
        index=0
    )
    
    st.markdown("### Enter 31 Morphometric Measurements")
    st.caption("📌 Meristic counts are integers. All other measurements in mm.")
    
    # ===============================
    # 31 INPUT FIELDS
    # ===============================
    
    # Column 1: Meristic (6)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📏 Meristic (6)")
        nd1 = st.number_input("ND1_Total", min_value=0.0, max_value=50.0, value=4.0, step=1.0, key="nd1_31")
        nd2 = st.number_input("ND2_Total", min_value=0.0, max_value=50.0, value=7.0, step=1.0, key="nd2_31")
        np_val = st.number_input("NP", min_value=0.0, max_value=50.0, value=14.0, step=1.0, key="np_31")
        nc = st.number_input("NC", min_value=0.0, max_value=50.0, value=14.0, step=1.0, key="nc_31")
        nv = st.number_input("NV_Total", min_value=0.0, max_value=50.0, value=6.0, step=1.0, key="nv_31")
        na = st.number_input("NA_Total", min_value=0.0, max_value=50.0, value=10.0, step=1.0, key="na_31")
    
    with col2:
        st.subheader("📐 Morphometric (4)")
        sl = st.number_input("SL (mm)", min_value=0.0, max_value=500.0, value=150.0, step=10.0, key="sl_31")
        pl = st.number_input("PL (mm)", min_value=0.0, max_value=300.0, value=40.0, step=5.0, key="pl_31")
        bh = st.number_input("BH (mm)", min_value=0.0, max_value=300.0, value=45.0, step=5.0, key="bh_31")
        hl = st.number_input("HL (mm)", min_value=0.0, max_value=300.0, value=40.0, step=5.0, key="hl_31")
    
    with col3:
        st.subheader("📐 Truss Network (21)")
        # Split truss into 3 sub-columns within col3
        truss_cols_1, truss_cols_2, truss_cols_3 = st.columns(3)
        
        with truss_cols_1:
            ab = st.number_input("A-B", min_value=0.0, max_value=200.0, value=8.0, step=1.0, key="ab_31")
            ac = st.number_input("A-C", min_value=0.0, max_value=200.0, value=30.0, step=1.0, key="ac_31")
            ad = st.number_input("A-D", min_value=0.0, max_value=200.0, value=25.0, step=1.0, key="ad_31")
            bc = st.number_input("B-C", min_value=0.0, max_value=200.0, value=25.0, step=1.0, key="bc_31")
            bd = st.number_input("B-D", min_value=0.0, max_value=200.0, value=20.0, step=1.0, key="bd_31")
            cd = st.number_input("C-D", min_value=0.0, max_value=200.0, value=25.0, step=1.0, key="cd_31")
            ce = st.number_input("C-E", min_value=0.0, max_value=300.0, value=45.0, step=1.0, key="ce_31")
        
        with truss_cols_2:
            cf = st.number_input("C-F", min_value=0.0, max_value=300.0, value=40.0, step=1.0, key="cf_31")
            de = st.number_input("D-E", min_value=0.0, max_value=300.0, value=55.0, step=1.0, key="de_31")
            df = st.number_input("D-F", min_value=0.0, max_value=300.0, value=30.0, step=1.0, key="df_31")
            ef = st.number_input("E-F", min_value=0.0, max_value=300.0, value=45.0, step=1.0, key="ef_31")
            eg = st.number_input("E-G", min_value=0.0, max_value=300.0, value=35.0, step=1.0, key="eg_31")
            eh = st.number_input("E-H", min_value=0.0, max_value=300.0, value=50.0, step=1.0, key="eh_31")
            fg = st.number_input("F-G", min_value=0.0, max_value=300.0, value=60.0, step=1.0, key="fg_31")
        
        with truss_cols_3:
            fh = st.number_input("F-H", min_value=0.0, max_value=300.0, value=45.0, step=1.0, key="fh_31")
            gh = st.number_input("G-H", min_value=0.0, max_value=300.0, value=35.0, step=1.0, key="gh_31")
            gi = st.number_input("G-I", min_value=0.0, max_value=200.0, value=35.0, step=1.0, key="gi_31")
            gj = st.number_input("G-J", min_value=0.0, max_value=200.0, value=40.0, step=1.0, key="gj_31")
            hi = st.number_input("H-I", min_value=0.0, max_value=200.0, value=45.0, step=1.0, key="hi_31")
            hj = st.number_input("H-J", min_value=0.0, max_value=200.0, value=35.0, step=1.0, key="hj_31")
            ij = st.number_input("I-J", min_value=0.0, max_value=200.0, value=18.0, step=1.0, key="ij_31")
    
    # ===============================
    # PREDICT BUTTON
    # ===============================
    
    if st.button("🔍 Predict Species", type="primary"):
        try:
            # Collect all 31 inputs in correct order
            input_values = [
                nd1, nd2, np_val, nc, nv, na,  # 6 meristic
                sl, pl, bh, hl,                # 4 morphometric
                ab, ac, ad, bc, bd, cd,        # 6 truss
                ce, cf, de, df, ef,            # 5 truss
                eg, eh, fg, fh, gh,            # 5 truss
                gi, gj, hi, hj, ij             # 5 truss
            ]
            
            # Convert to numpy array with shape (1, 31)
            input_array = np.array(input_values, dtype=np.float64).reshape(1, -1)
            
            # Standardize
            input_scaled = scaler.transform(input_array)
            
            # Select model
            if "GWO" in model_choice:
                model = models['gwo']
                model_name = "ANN-GWO"
            elif "PSO" in model_choice:
                model = models['pso']
                model_name = "ANN-PSO"
            elif "GA" in model_choice:
                model = models['ga']
                model_name = "ANN-GA"
            else:
                model = models['ann']
                model_name = "ANN"
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            predicted_species = label_encoder.inverse_transform([prediction])[0]
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = np.max(probabilities) * 100
            
            # ===============================
            # DISPLAY RESULTS WITH IMAGE
            # ===============================
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Create columns for image and info
            col_img, col_info = st.columns([1, 2])
            
            with col_img:
                # Try to load species image
                image_path = f"images/{predicted_species.lower().replace(' ', '_')}.jpg"
                try:
                    if os.path.exists(image_path):
                        img = Image.open(image_path)
                        st.image(img, caption=predicted_species, use_container_width=True)
                    else:
                        # Fallback: try other extensions
                        alt_paths = [
                            f"images/{predicted_species.lower().replace(' ', '_')}.png",
                            f"images/{predicted_species.lower().replace(' ', '_')}.jpeg"
                        ]
                        found = False
                        for alt in alt_paths:
                            if os.path.exists(alt):
                                img = Image.open(alt)
                                st.image(img, caption=predicted_species, use_container_width=True)
                                found = True
                                break
                        if not found:
                            st.markdown(f"<h1 style='font-size:80px; text-align:center;'>🐟</h1>", unsafe_allow_html=True)
                            st.caption(f"Image not found for {predicted_species}")
                except Exception as e:
                    st.markdown(f"<h1 style='font-size:80px; text-align:center;'>🐟</h1>", unsafe_allow_html=True)
                    st.caption(f"Image not available")
            
            with col_info:
                st.markdown(f"""
                <div style="padding: 20px; background-color: #f0f8ff; border-radius: 10px; border: 2px solid #1e90ff;">
                    <h2 style="color: #1e90ff;">🐟 {predicted_species}</h2>
                    <p><b>Confidence:</b> {confidence:.1f}%</p>
                    <p><b>Model:</b> {model_name}</p>
                    <p><b>Data Mode:</b> {data_mode}</p>
                    <p><b>Features:</b> 31 (6 Meristic + 4 Morphometric + 21 Truss)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(int(confidence))
            
            # ===============================
            # PROBABILITY DISTRIBUTION
            # ===============================
            
            st.subheader("📊 Species Probabilities")
            
            prob_df = pd.DataFrame({
                'Species': label_encoder.classes_,
                'Probability (%)': probabilities * 100
            }).sort_values('Probability (%)', ascending=False)
            
            st.bar_chart(prob_df.set_index('Species'))
            
            # ===============================
            # TRUSS DETAILS (21 measurements)
            # ===============================
            
            with st.expander("📐 Truss Network Details (21 measurements)"):
                truss_data = {
                    'Measurement': ['A-B', 'A-C', 'A-D', 'B-C', 'B-D', 'C-D', 
                                    'C-E', 'C-F', 'D-E', 'D-F', 'E-F',
                                    'E-G', 'E-H', 'F-G', 'F-H', 'G-H',
                                    'G-I', 'G-J', 'H-I', 'H-J', 'I-J'],
                    'Value (mm)': [ab, ac, ad, bc, bd, cd, ce, cf, de, df, ef,
                                   eg, eh, fg, fh, gh, gi, gj, hi, hj, ij]
                }
                truss_df = pd.DataFrame(truss_data)
                st.dataframe(truss_df, use_container_width=True)
            
            # ===============================
            # DEBUG INFO
            # ===============================
            
            with st.expander("🔍 Debug Information"):
                st.write("**Input Features (31 values):**", input_values)
                st.write("**Input Shape:**", input_array.shape)
                st.write("**Prediction Class Index:**", prediction)
                st.write("**Species Classes:**", list(label_encoder.classes_))
                st.write("**Probabilities Array:**", probabilities)
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("Please check that all input values are valid numbers.")
            st.code(f"Error details: {str(e)}")

else:
    st.error("❌ Models not loaded. Please ensure all .pkl files are uploaded to GitHub.")
    st.info("""
    Required files for 31 features:
    - ann_model_31features.pkl
    - pso_model_31features.pkl
    - ga_model_31features.pkl
    - gwo_model_31features.pkl
    - scaler_31features.pkl
    - label_encoder_31features.pkl
    - feature_names_31features.pkl
    
    Also upload:
    - requirements.txt
    - runtime.txt
    - images/ folder with species images
    
    Then click 'Redeploy' on Streamlit Cloud.
    """)

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>🐟 Mugilidae Fish Classification System | 31 Features (6 Meristic + 4 Morphometric + 21 Truss)</p>
<p>FYP Project | Universiti Malaysia Terengganu</p>
</div>
""", unsafe_allow_html=True)
