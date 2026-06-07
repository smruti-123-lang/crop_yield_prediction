import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Load models & encoders ────────────────────────────────────────
import os

# Get the folder where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load models & encoders ────────────────────────────────────────
lr_model    = pickle.load(open(os.path.join(BASE_DIR, 'lr_model.pkl'),    'rb'))
lasso_model = pickle.load(open(os.path.join(BASE_DIR, 'lasso_model.pkl'), 'rb'))
ridge_model = pickle.load(open(os.path.join(BASE_DIR, 'ridge_model.pkl'), 'rb'))
rf_model    = pickle.load(open(os.path.join(BASE_DIR, 'rf_model.pkl'),    'rb'))

le_crop     = pickle.load(open(os.path.join(BASE_DIR, 'le_crop.pkl'),   'rb'))
le_state    = pickle.load(open(os.path.join(BASE_DIR, 'le_state.pkl'),  'rb'))
le_season   = pickle.load(open(os.path.join(BASE_DIR, 'le_season.pkl'), 'rb'))
# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title='Crop Yield Predictor', page_icon='🌾', layout='centered')
st.title('🌾 Crop Yield Predictor')
st.markdown('Fill in the details below to predict crop yield using 4 models.')

# ── Input form ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    crop    = st.selectbox('Crop',   sorted(le_crop.classes_))
    state   = st.selectbox('State',  sorted(le_state.classes_))
    season  = st.selectbox('Season', sorted(le_season.classes_))
    year    = st.slider('Crop Year', 1997, 2030, 2020)

with col2:
    area        = st.number_input('Area (hectares)',       min_value=0.1,  value=1000.0)
    production  = st.number_input('Production (tonnes)',   min_value=0.1,  value=1500.0)
    rainfall    = st.number_input('Annual Rainfall (mm)',  min_value=0.0,  value=1200.0)
    fertilizer  = st.number_input('Fertilizer (kg)',       min_value=0.0,  value=50000.0)
    pesticide   = st.number_input('Pesticide (kg)',        min_value=0.0,  value=1000.0)

# ── Predict button ────────────────────────────────────────────────
if st.button('🔍 Predict Yield', use_container_width=True):

    # Encode categoricals
    crop_enc   = le_crop.transform([crop])[0]
    state_enc  = le_state.transform([state])[0]
    season_enc = le_season.transform([season])[0]

    # Feature engineering — same as training
    yield_per_area      = production / (area + 1)
    fertilizer_per_area = fertilizer / (area + 1)
    pesticide_per_area  = pesticide  / (area + 1)
    production_per_rain = production / (rainfall + 1)
    log_area            = np.log1p(area)
    log_production      = np.log1p(production)
    log_fertilizer      = np.log1p(fertilizer)
    log_pesticide       = np.log1p(pesticide)

    # Build input row — same order as training features
    input_data = pd.DataFrame([[
        crop_enc, state_enc, season_enc, year,
        log_area, log_production, log_fertilizer, log_pesticide,
        yield_per_area, fertilizer_per_area, pesticide_per_area,
        production_per_rain, rainfall
    ]], columns=[
        'Crop_enc', 'State_enc', 'Season_enc', 'Crop_Year',
        'log_area', 'log_production', 'log_fertilizer', 'log_pesticide',
        'yield_per_area', 'fertilizer_per_area', 'pesticide_per_area',
        'production_per_rain', 'Annual_Rainfall'
    ])

    # Predict — models trained on log_yield so inverse with expm1
    results = {
        'Linear Regression' : np.expm1(lr_model.predict(input_data)[0]),
        'Lasso'             : np.expm1(lasso_model.predict(input_data)[0]),
        'Ridge'             : np.expm1(ridge_model.predict(input_data)[0]),
        'Random Forest'     : np.expm1(rf_model.predict(input_data)[0]),
    }

    # ── Display results ───────────────────────────────────────────
    st.markdown('---')
    st.subheader('📊 Predicted Yield (tonnes/hectare)')

    c1, c2, c3, c4 = st.columns(4)
    for col, (name, val) in zip([c1, c2, c3, c4], results.items()):
        col.metric(label=name, value=f'{val:.2f}')

    # Best model highlight
    st.success(f"✅ Random Forest predicts: **{results['Random Forest']:.2f} tonnes/hectare**")