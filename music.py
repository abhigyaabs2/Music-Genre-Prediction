import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="Music Genre Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Load model and vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load('classifier_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first.")
        st.stop()

model, tfidf = load_models()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return ' '.join(words)

# Prediction function
def predict_genre(lyrics):
    processed = preprocess_text(lyrics)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    genres = model.classes_
    genre_probs = dict(zip(genres, probabilities))
    return prediction, genre_probs, processed

# Genre configurations
genre_config = {
    'Rock': {'emoji': '🎸', 'color': '#E74C3C', 'gradient': 'linear-gradient(135deg, #E74C3C 0%, #C0392B 100%)'},
    'Pop': {'emoji': '🎤', 'color': '#F39C12', 'gradient': 'linear-gradient(135deg, #F39C12 0%, #E67E22 100%)'},
    'Hip-Hop': {'emoji': '🎵', 'color': '#9B59B6', 'gradient': 'linear-gradient(135deg, #9B59B6 0%, #8E44AD 100%)'},
    'Country': {'emoji': '🤠', 'color': '#27AE60', 'gradient': 'linear-gradient(135deg, #27AE60 0%, #229954 100%)'},
    'Electronic': {'emoji': '🎧', 'color': '#3498DB', 'gradient': 'linear-gradient(135deg, #3498DB 0%, #2980B9 100%)'}
}

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1, h2, h3 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    .prediction-card {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        text-align: center;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .genre-badge {
        display: inline-block;
        padding: 20px 50px;
        border-radius: 50px;
        font-size: 36px;
        font-weight: 700;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .confidence-bar {
        height: 8px;
        background: #ecf0f1;
        border-radius: 10px;
        overflow: hidden;
        margin: 20px 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }
    
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .example-btn {
        background: rgba(255,255,255,0.2);
        border: 2px solid white;
        border-radius: 10px;
        padding: 15px;
        color: white;
        cursor: pointer;
        transition: all 0.3s;
        margin: 5px 0;
        text-align: left;
    }
    
    .example-btn:hover {
        background: rgba(255,255,255,0.3);
        transform: translateX(10px);
    }
    
    .stTextArea label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 18px !important;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .info-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }
    
    .stats-number {
        font-size: 32px;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .stats-label {
        font-size: 14px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Header
col_h1, col_h2, col_h3 = st.columns([1, 2, 1])
with col_h2:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 60px; margin-bottom: 0;'>🎵 Music Genre Predictor</h1>
            <p style='color: white; font-size: 20px; margin-top: 10px;'>
                Powered by AI & Machine Learning
            </p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 80px;'>🎼</div>
            <h2>About This App</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-card'>
            <p>🤖 <strong>AI-Powered Classification</strong></p>
            <p>Uses Random Forest and TF-IDF to analyze lyrics and predict music genres.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎭 Supported Genres")
    for genre, config in genre_config.items():
        st.markdown(f"**{config['emoji']} {genre}**")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style='text-align: center;'>
                <div class='stats-number'>880+</div>
                <div class='stats-label'>Samples</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style='text-align: center;'>
                <div class='stats-number'>300</div>
                <div class='stats-label'>Features</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
        <div class='info-card'>
            <p><strong>💡 How it works:</strong></p>
            <p>1️⃣ Text preprocessing<br>
            2️⃣ TF-IDF vectorization<br>
            3️⃣ Random Forest prediction<br>
            4️⃣ Probability analysis</p>
        </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 📝 Enter Your Lyrics")
    
    lyrics_input = st.text_area(
        "Type or paste song lyrics below:",
        height=250,
        placeholder="Example: electric guitar solo rock and roll energy power loud drums amplifier...\n\nOr try clicking an example genre on the right! →",
        key="lyrics_area"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        predict_btn = st.button("🎯 Predict Genre", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    with col_btn3:
        random_btn = st.button("🎲 Random", use_container_width=True)

with col_right:
    st.markdown("### 🎼 Quick Examples")
    
    examples = {
        'Rock': "electric guitar solo rock and roll energy power loud drums amplifier distortion rebel freedom stage band",
        'Pop': "love dancing baby tonight party celebration happiness smile forever summer sunshine beautiful feeling amazing",
        'Hip-Hop': "rap flow beat street hustle bars verses microphone rhyme culture urban swagger freestyle cipher underground",
        'Country': "country road truck farm hometown small town whiskey heartbreak cowboy boots sunset memories simple honest",
        'Electronic': "synthesizer techno beat drop bass edm festival rave house dance club dj mixing production"
    }
    
    for genre, example in examples.items():
        config = genre_config[genre]
        if st.button(f"{config['emoji']} {genre}", key=f"btn_{genre}", use_container_width=True):
            st.session_state.selected_example = example
            st.rerun()

# Handle example selection
if 'selected_example' in st.session_state:
    lyrics_input = st.session_state.selected_example
    del st.session_state.selected_example
    st.rerun()

# Handle random button
if random_btn:
    import random
    lyrics_input = random.choice(list(examples.values()))
    st.rerun()

# Handle clear button
if clear_btn:
    lyrics_input = ""
    st.rerun()

# Prediction
if predict_btn and lyrics_input.strip():
    with st.spinner("🔮 Analyzing your lyrics..."):
        prediction, probabilities, processed = predict_genre(lyrics_input)
        config = genre_config[prediction]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🎊 Prediction Results")
        
        # Main prediction card
        col_res1, col_res2, col_res3 = st.columns([1, 3, 1])
        with col_res2:
            confidence = probabilities[prediction] * 100
            st.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='color: #7f8c8d; margin-bottom: 10px;'>Predicted Genre</h3>
                    <div class='genre-badge' style='background: {config["gradient"]};'>
                        {config['emoji']} {prediction}
                    </div>
                    <h2 style='color: #2c3e50; margin-top: 20px;'>{confidence:.1f}% Confidence</h2>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width: {confidence}%; background: {config["color"]};'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Probability distribution chart
        st.markdown("### 📊 Genre Probability Distribution")
        
        sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure()
        
        colors = [genre_config[g]['color'] for g in sorted_probs.keys()]
        
        fig.add_trace(go.Bar(
            x=list(sorted_probs.keys()),
            y=[p * 100 for p in sorted_probs.values()],
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.2)', width=2)
            ),
            text=[f"{p*100:.1f}%" for p in sorted_probs.values()],
            textposition='outside',
            textfont=dict(size=14, color='white', family='Poppins'),
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(255,255,255,0.1)',
            font=dict(size=14, color='white', family='Poppins'),
            height=400,
            showlegend=False,
            xaxis=dict(
                title="Genre",
                titlefont=dict(size=16, color='white'),
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title="Probability (%)",
                titlefont=dict(size=16, color='white'),
                tickfont=dict(size=14, color='white'),
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.2)'
            ),
            margin=dict(t=20, b=60, l=60, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed probability cards
        st.markdown("### 🎯 Detailed Analysis")
        prob_cols = st.columns(5)
        
        for idx, (genre, prob) in enumerate(sorted_probs.items()):
            with prob_cols[idx]:
                config = genre_config[genre]
                rank = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "📊"
                st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-size: 40px; margin-bottom: 10px;'>{config['emoji']}</div>
                        <div style='font-size: 16px; font-weight: 600; color: {config["color"]};'>{genre}</div>
                        <div style='font-size: 28px; font-weight: 700; color: #2c3e50; margin: 10px 0;'>{prob*100:.1f}%</div>
                        <div style='font-size: 20px;'>{rank}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("<br>", unsafe_allow_html=True)
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            with st.expander("🔍 Preprocessed Text", expanded=False):
                st.code(processed, language=None)
                st.caption("Text after preprocessing: lowercase, stopwords removed, and lemmatized")
        
        with col_ins2:
            with st.expander("📈 Confidence Analysis", expanded=False):
                if confidence > 80:
                    st.success(f"✅ **Very High Confidence** - The model is {confidence:.1f}% certain this is {prediction}")
                elif confidence > 60:
                    st.info(f"✓ **Good Confidence** - Strong indicators suggest this is {prediction}")
                elif confidence > 40:
                    st.warning(f"⚠️ **Moderate Confidence** - Some characteristics of {prediction}, but could be mixed genre")
                else:
                    st.error(f"❌ **Low Confidence** - Difficult to classify, may have mixed genre characteristics")
                
                top_2 = list(sorted_probs.items())[:2]
                diff = (top_2[0][1] - top_2[1][1]) * 100
                st.caption(f"Difference from second choice ({top_2[1][0]}): {diff:.1f}%")

elif predict_btn:
    st.warning("⚠️ Please enter some lyrics to analyze!")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: white; padding: 30px;'>
        <hr style='border: 1px solid rgba(255,255,255,0.2); margin: 30px 0;'>
        <h3>🎵 Music Genre Predictor</h3>
        <p style='font-size: 16px;'>Built with ❤️ using Streamlit • Scikit-learn • NLTK</p>
        <p style='font-size: 14px; opacity: 0.8;'>Random Forest Classifier trained on 880+ diverse lyrics samples</p>
        <p style='font-size: 12px; opacity: 0.6; margin-top: 20px;'>
            🎸 Rock • 🎤 Pop • 🎵 Hip-Hop • 🤠 Country • 🎧 Electronic
        </p>
    </div>
""", unsafe_allow_html=True)