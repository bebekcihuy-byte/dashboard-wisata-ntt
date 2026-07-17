
# -*- coding: utf-8 -*-
"""
Dashboard Analisis Sentimen & Topic Modeling
Desa Wae Rebo vs Taman Nasional Komodo
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Dashboard Wisata NTT",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #171863 0%, #2d2e8a 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-title {
        color: #171863;
        border-bottom: 3px solid #fca919;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    try:
        with open("dashboard_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

data = load_data()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🏔️ Navigation")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "🏠 Beranda",
        "📊 Analisis Sentimen",
        "🔢 Klasterisasi",
        "📝 Topic Modeling",
        "📈 Trending Topic"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Analisis Wisata NTT\nDesa Wae Rebo vs Taman Nasional Komodo")

# ============================================================
# HALAMAN BERANDA
# ============================================================
if menu == "🏠 Beranda":
    st.markdown("""
    <div class="main-header">
        <h1>🏞️ Dashboard Analisis Wisata NTT</h1>
        <p style="font-size: 1.2rem;">Perbandingan Destinasi: Desa Wae Rebo vs Taman Nasional Komodo</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analisis Sentimen</h3>
            <p>Perbandingan sentimen positif, negatif, dan netral dari ulasan pengunjung</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔢 Klasterisasi</h3>
            <p>Pengelompokan ulasan menggunakan algoritma K-Means</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📝 Topic Modeling</h3>
            <p>Ekstraksi topik utama menggunakan LDA (Latent Dirichlet Allocation)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Tentang Dashboard
    Dashboard ini menampilkan hasil analisis teks terhadap ulasan pengunjung di Google Maps 
    untuk dua destinasi wisata populer di Nusa Tenggara Timur:
    
    - **Desa Wae Rebo** - Desa adat yang terletak di ketinggian 1.200 mdpl di Manggarai, Flores
    - **Taman Nasional Komodo** - Habitat asli komodo dan kekayaan bahari di Labuan Bajo
    
    ### 🔧 Metodologi
    1. **Preprocessing**: Cleaning, Case Folding, Normalisasi, Tokenisasi, Stopword Removal, Stemming
    2. **Analisis Sentimen**: Lexicon-based (InSet)
    3. **Klasifikasi**: SVM, Random Forest, Naive Bayes
    4. **Klasterisasi**: K-Means dengan TF-IDF + SVD
    5. **Topic Modeling**: LDA (Latent Dirichlet Allocation)
    """)

# ============================================================
# HALAMAN ANALISIS SENTIMEN
# ============================================================
elif menu == "📊 Analisis Sentimen":
    st.title("📊 Analisis Sentimen")
    st.markdown("<div class='section-title'><h2>Distribusi Sentimen Ulasan Pengunjung</h2></div>", unsafe_allow_html=True)
    
    # Cek apakah data sentimen tersedia
    if "sentimen" in data:
        sentimen_data = data["sentimen"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Desa Wae Rebo")
            if "Desa Wae Rebo" in sentimen_data:
                df_wr = pd.DataFrame(sentimen_data["Desa Wae Rebo"])
                fig = px.pie(df_wr, values='Jumlah', names='Sentimen', 
                           color_discrete_sequence=['#059669', '#DC2626', '#6B7280'],
                           hole=0.4)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_wr, use_container_width=True)
            else:
                st.info("Data sentimen Desa Wae Rebo belum tersedia")
        
        with col2:
            st.subheader("Taman Nasional Komodo")
            if "Taman Nasional Komodo" in sentimen_data:
                df_komodo = pd.DataFrame(sentimen_data["Taman Nasional Komodo"])
                fig = px.pie(df_komodo, values='Jumlah', names='Sentimen',
                           color_discrete_sequence=['#059669', '#DC2626', '#6B7280'],
                           hole=0.4)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_komodo, use_container_width=True)
            else:
                st.info("Data sentimen Taman Nasional Komodo belum tersedia")
    else:
        st.warning("Data sentimen belum tersedia. Pastikan file JSON berisi data sentimen.")
        
        # Tampilkan contoh visualisasi dummy
        st.markdown("---")
        st.subheader("📌 Contoh Visualisasi (Demo)")
        
        dummy_data = pd.DataFrame({
            'Sentimen': ['Positif', 'Netral', 'Negatif'],
            'Desa Wae Rebo': [120, 45, 15],
            'Taman Nasional Komodo': [180, 60, 25]
        })
        
        fig = px.bar(dummy_data, x='Sentimen', y=['Desa Wae Rebo', 'Taman Nasional Komodo'],
                    barmode='group', color_discrete_sequence=['#171863', '#fca919'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# HALAMAN KLASTERISASI
# ============================================================
elif menu == "🔢 Klasterisasi":
    st.title("🔢 Klasterisasi K-Means")
    st.markdown("<div class='section-title'><h2>Pengelompokan Ulasan Berdasarkan Kesamaan</h2></div>", unsafe_allow_html=True)
    
    if "klaster" in data:
        klaster_data = data["klaster"]
        
        tabs = st.tabs(list(klaster_data.keys()))
        
        for idx, lokasi in enumerate(klaster_data.keys()):
            with tabs[idx]:
                st.subheader(f"📍 {lokasi}")
                loc_data = klaster_data[lokasi]
                
                # Metrik evaluasi
                if "metrik" in loc_data:
                    metrik = loc_data["metrik"]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Silhouette Score", metrik.get("silhouette", "N/A"))
                    col2.metric("Davies-Bouldin", metrik.get("davies_bouldin", "N/A"))
                    col3.metric("Jumlah Klaster", metrik.get("n_clusters", "N/A"))
                
                # Scatter plot (jika ada koordinat)
                if "scatter" in loc_data:
                    scatter = loc_data["scatter"]
                    fig = px.scatter(
                        x=scatter["x"], y=scatter["y"],
                        color=scatter["labels"],
                        title=f"Visualisasi Klaster - {lokasi}",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Kata dominan per klaster
                if "kata_dominan" in loc_data:
                    st.subheader("Kata Dominan per Klaster")
                    df_kata = pd.DataFrame(loc_data["kata_dominan"])
                    st.dataframe(df_kata, use_container_width=True)
    else:
        st.warning("Data klasterisasi belum tersedia.")
        st.markdown("""
        ### 📌 Informasi Klasterisasi
        Klasterisasi dilakukan menggunakan:
        - **Algoritma**: K-Means
        - **Fitur**: TF-IDF (1000 fitur)
        - **Reduksi Dimensi**: Truncated SVD (2 komponen)
        - **Penentuan K**: Elbow Method + Silhouette Score
        """)

# ============================================================
# HALAMAN TOPIC MODELING
# ============================================================
elif menu == "📝 Topic Modeling":
    st.title("📝 Topic Modeling (LDA)")
    st.markdown("<div class='section-title'><h2>Ekstraksi Topik Utama dari Ulasan</h2></div>", unsafe_allow_html=True)
    
    if "topik" in data:
        topik_data = data["topik"]
        
        tabs = st.tabs(list(topik_data.keys()))
        
        for idx, lokasi in enumerate(topik_data.keys()):
            with tabs[idx]:
                st.subheader(f"📍 {lokasi}")
                loc_data = topik_data[lokasi]
                
                # Pie chart distribusi topik
                if "labels" in loc_data and "values" in loc_data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        df_topik = pd.DataFrame({
                            "Topik": loc_data["labels"],
                            "Jumlah": loc_data["values"]
                        })
                        fig = px.pie(df_topik, values='Jumlah', names='Topik',
                                   hole=0.4,
                                   color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.dataframe(df_topik, use_container_width=True)
                
                # Bar chart kata per topik
                if "keywords_detail" in loc_data:
                    st.subheader("Kata Kunci per Topik")
                    keywords = loc_data["keywords_detail"]
                    
                    n_topik = len(keywords)
                    n_cols = min(3, n_topik)
                    n_rows = (n_topik + n_cols - 1) // n_cols
                    
                    for topic_idx in range(n_topik):
                        topic_key = str(topic_idx)
                        if topic_key in keywords:
                            words_vals = keywords[topic_key]
                            df_words = pd.DataFrame(words_vals, columns=["Kata", "Bobot"])
                            df_words = df_words.sort_values("Bobot", ascending=True)
                            
                            fig = px.barh(df_words, x="Bobot", y="Kata",
                                        title=f"Topik {topic_idx}: {loc_data['labels'][topic_idx]}",
                                        color="Bobot",
                                        color_continuous_scale="Blues")
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data topic modeling belum tersedia.")
        st.markdown("""
        ### 📌 Informasi Topic Modeling
        Topic modeling dilakukan menggunakan:
        - **Algoritma**: LDA (Latent Dirichlet Allocation)
        - **Fitur**: Bag of Words (500 fitur)
        - **Penentuan K**: Perplexity Score
        - **Auto-labeling**: Berdasarkan kata kunci dominan
        """)

# ============================================================
# HALAMAN TRENDING TOPIC
# ============================================================
elif menu == "📈 Trending Topic":
    st.title("📈 Trending Topic")
    st.markdown("<div class='section-title'><h2>Tren Topik dari Waktu ke Waktu</h2></div>", unsafe_allow_html=True)
    
    if "trending" in data:
        trending_data = data["trending"]
        
        for lokasi in trending_data.keys():
            st.subheader(f"📍 {lokasi}")
            loc_data = trending_data[lokasi]
            
            if "bulan" in loc_data and "topik" in loc_data:
                df_trend = pd.DataFrame({
                    "Bulan": loc_data["bulan"]
                })
                
                for topik_name, values in loc_data["topik"].items():
                    df_trend[topik_name] = values
                
                fig = px.line(df_trend, x="Bulan", y=list(loc_data["topik"].keys()),
                            markers=True,
                            title=f"Trending Topic - {lokasi}",
                            color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(
                    height=400,
                    xaxis_title="Bulan",
                    yaxis_title="Jumlah Review",
                    legend_title="Topik"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tampilkan data tabel
                with st.expander("📊 Lihat Data Tabel"):
                    st.dataframe(df_trend, use_container_width=True)
            
            st.markdown("---")
    else:
        st.warning("Data trending topic belum tersedia.")
        st.markdown("""
        ### 📌 Informasi Trending
        Trending topic dianalisis berdasarkan:
        - **Parsing Tanggal**: Konversi teks relatif ("2 minggu lalu") ke tanggal absolut
        - **Agregasi**: Jumlah review per topik per bulan
        - **Visualisasi**: Line chart untuk melihat tren
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("© 2024 Dashboard Analisis Wisata NTT | Dibuat dengan Streamlit")
