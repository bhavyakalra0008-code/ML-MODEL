import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Set page configurations
st.set_page_config(
    page_title="AI Gender Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d111d;
        color: #f1f5f9;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header background */
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid #1f2937;
    }
    
    /* Styled container cards */
    .card {
        background: linear-gradient(145deg, #141b2e 0%, #0d1222 100%);
        border: 1px solid #1e294b;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .card:hover {
        border-color: #4f46e5;
        box-shadow: 0 15px 35px rgba(79, 70, 229, 0.15);
        transform: translateY(-2px);
    }
    
    /* Accent text and headers */
    .gradient-title {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.05rem;
        margin-bottom: 4px;
    }
    .gradient-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 300;
        margin-bottom: 28px;
    }
    
    /* Metric styling */
    .metric-box {
        background: #141b2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #1e294b;
    }
    
    /* Result Display */
    .result-container {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 10px;
        transition: all 0.4s ease;
    }
    .female-result {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(244, 63, 94, 0.02) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        box-shadow: 0 0 25px rgba(244, 63, 94, 0.1);
    }
    .male-result {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.02) 100%);
        border: 1px solid rgba(59, 130, 246, 0.4);
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.1);
    }
    
    .result-emoji {
        font-size: 4rem;
        margin-bottom: 12px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.08); }
        100% { transform: scale(1); }
    }
    
    .result-label {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .result-gender-female {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f43f5e;
    }
    .result-gender-male {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .confidence-val {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 10px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #4b5563;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    # Clean string spaces
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    return df

# Load dataset
csv_filename = "Dummy Dataset 1.csv"
df = load_data(csv_filename)

if df is None:
    st.error(f"🚨 Dataset not found! Make sure `{csv_filename}` is present in the workspace directory.")
    st.stop()

# Cache the ML Model Pipeline
@st.cache_resource
def train_model(data):
    # Split features and target
    feature_cols = ['Favorite Color', 'Favorite Music Genre', 'Favorite Beverage', 'Favorite Soft Drink']
    X = data[feature_cols]
    y = data['Gender']
    
    # Define Encoder & Classifier
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    
    pipeline.fit(X, y)
    
    # Calculate accuracy
    preds = pipeline.predict(X)
    acc = accuracy_score(y, preds)
    return pipeline, acc

# Train the model
model_pipeline, model_accuracy = train_model(df)

# Sidebar - Project Overview
with st.sidebar:
    st.markdown("### 🧬 AI Model Statistics")
    st.markdown(f"**Model Type**: Random Forest Classifier")
    st.markdown(f"**Training Dataset size**: {len(df)} samples")
    st.markdown(f"**Model Accuracy on Training Set**: {model_accuracy:.1%}")
    
    st.markdown("---")
    st.markdown("### 📋 Features Analyzed")
    st.write("- **Favorite Color**")
    st.write("- **Favorite Music Genre**")
    st.write("- **Favorite Beverage**")
    st.write("- **Favorite Soft Drink**")
    
    st.markdown("---")
    st.markdown("Designed for pairing exploration and local testing.")

# Main Page Header
st.markdown('<div class="gradient-title">🧬 Gender Predictor AI</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subtitle">A high-fidelity machine learning interface predicting gender identity profiles from beverage, soft drink, music, and color preferences.</div>', unsafe_allow_html=True)

# Layout division
col1, col2 = st.columns([1, 1], gap="large")

# Extract categories dynamically for form selectors
color_options = sorted(df['Favorite Color'].unique())
genre_options = sorted(df['Favorite Music Genre'].unique())
beverage_options = sorted(df['Favorite Beverage'].unique())
soft_drink_options = sorted(df['Favorite Soft Drink'].unique())

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎨 Enter Preferences Profile")
    st.write("Specify color, music, and beverage choices to test prediction:")
    
    # Widgets
    selected_color = st.selectbox("Favorite Color", color_options)
    selected_genre = st.selectbox("Favorite Music Genre", genre_options)
    selected_beverage = st.selectbox("Favorite Beverage", beverage_options)
    selected_soft_drink = st.selectbox("Favorite Soft Drink", soft_drink_options)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
    st.subheader("🔮 AI Prediction Result")
    
    # Create single input dataframe matching features
    input_data = pd.DataFrame([{
        'Favorite Color': selected_color,
        'Favorite Music Genre': selected_genre,
        'Favorite Beverage': selected_beverage,
        'Favorite Soft Drink': selected_soft_drink
    }])
    
    # Run Prediction
    prediction = model_pipeline.predict(input_data)[0]
    probabilities = model_pipeline.predict_proba(input_data)[0]
    classes = model_pipeline.classes_
    
    # Map confidence score to predicted class
    pred_index = np.where(classes == prediction)[0][0]
    confidence = probabilities[pred_index]
    
    # Beautiful prediction card
    if prediction == 'F':
        gender_full = "Female"
        emoji = "👧"
        theme_class = "female-result"
        gender_style_class = "result-gender-female"
        progress_color = "#f43f5e"
    else:
        gender_full = "Male"
        emoji = "👦"
        theme_class = "male-result"
        gender_style_class = "result-gender-male"
        progress_color = "#3b82f6"
        
    st.markdown(f"""
    <div class="result-container {theme_class}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">Predicted Profile</div>
        <div class="{gender_style_class}">{gender_full}</div>
        <div class="confidence-val">Confidence: {confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar showing probability split
    st.markdown("<br>", unsafe_allow_html=True)
    f_prob = probabilities[np.where(classes == 'F')[0][0]]
    m_prob = probabilities[np.where(classes == 'M')[0][0]]
    
    st.write("**Probability breakdown:**")
    col_f, col_m = st.columns(2)
    col_f.metric("Female Likelihood", f"{f_prob:.1%}")
    col_m.metric("Male Likelihood", f"{m_prob:.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Interactive Data Insights section
st.markdown("---")
with st.expander("📊 Dataset Analytics & Profile Insights", expanded=False):
    st.markdown("### Profile distributions in the training dataset")
    
    # Analytics layout
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    f_count = len(df[df['Gender'] == 'F'])
    m_count = len(df[df['Gender'] == 'M'])
    
    with col_metric1:
        st.markdown(f"""
        <div class="metric-box">
            <h4 style="color:#94a3b8; margin:0;">Total Samples</h4>
            <p style="font-size:2rem; font-weight:700; margin:5px 0 0 0; color:#818cf8;">{len(df)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_metric2:
        st.markdown(f"""
        <div class="metric-box">
            <h4 style="color:#94a3b8; margin:0;">Female Profiles</h4>
            <p style="font-size:2rem; font-weight:700; margin:5px 0 0 0; color:#f43f5e;">{f_count} ({f_count/len(df):.1%})</p>
        </div>
        """, unsafe_allow_html=True)
    with col_metric3:
        st.markdown(f"""
        <div class="metric-box">
            <h4 style="color:#94a3b8; margin:0;">Male Profiles</h4>
            <p style="font-size:2rem; font-weight:700; margin:5px 0 0 0; color:#3b82f6;">{m_count} ({m_count/len(df):.1%})</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Custom charts
    st.markdown("#### Favorite Color Distribution by Gender")
    color_gender = df.groupby(['Favorite Color', 'Gender']).size().unstack(fill_value=0)
    st.bar_chart(color_gender)
    
    st.markdown("#### Favorite Music Genre Distribution by Gender")
    music_gender = df.groupby(['Favorite Music Genre', 'Gender']).size().unstack(fill_value=0)
    st.bar_chart(music_gender)
    
    st.markdown("#### Raw Dataset Table")
    st.dataframe(df, use_container_width=True)

st.markdown('<div class="footer">AI Gender Predictor App • Running Locally on Streamlit</div>', unsafe_allow_html=True)
