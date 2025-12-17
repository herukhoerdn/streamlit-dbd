import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Prediksi DBD",
    page_icon="🦟",
    layout="wide"
)
st.markdown("""
<style>
body {
    background-color: #F5F7FB;
}

/* Main container */
.block-container {
    padding: 2.5rem;
    background-color: #F5F7FB;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E0E7FF;
    padding: 1.5rem;
}

/* Card */
.card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

/* Titles */
h1, h2, h3 {
    color: #1E3A8A;
}

/* Button */
.stButton>button {
    background-color: #2563EB;
    color: white;
    padding: 10px 26px;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1D4ED8;
}
</style>
""", unsafe_allow_html=True)
st.sidebar.title(" Input Data")

hujan = st.sidebar.number_input("Curah Hujan (mm)", 25, 3000, value=25)
kelembaban = st.sidebar.number_input("Kelembaban (%)", 10, 100, value=10)
suhu = st.sidebar.number_input("Suhu Rata-rata (°C)", 15, 100, value=15)
pddk = st.sidebar.number_input("Kepadatan Penduduk", 60, 30000, value=60)
banjir = st.sidebar.number_input("Jumlah Banjir", 0, 200, value=0)

st.markdown("""
<div class="card">
<h1>🦟 Prediksi Kasus Positif DBD</h1>
<p>Aplikasi prediksi kasus DBD berbasis Machine Learning untuk analisis faktor lingkungan</p>
</div>
""", unsafe_allow_html=True)
if st.button(" Prediksi Sekarang"):
    model = pickle.load(open("regression_dbd.pkl", "rb"))

    input_data = np.array([[hujan, kelembaban, suhu, pddk, banjir]])
    prediction = max(0, model.predict(input_data)[0])
    prediction_int = int(round(prediction))
    st.markdown(f"""
    <div class="card">
        <h2>📊 Hasil Prediksi</h2>
        <h1 style="color:#DC2626;">{prediction_int} Kasus</h1>
        <p>Perkiraan jumlah kasus DBD berdasarkan input faktor lingkungan</p>
    </div>
    """, unsafe_allow_html=True)
    df = pd.read_csv("positif_dbd1.csv")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(" Visualisasi 3D Faktor Penyebab DBD")

    fig_3d = px.scatter_3d(
        df,
        x="kepadatan_penduduk",
        y="curah_hujan_mm",
        z="jumlah_banjir",
        color="jumlah_kasus_positif_dbd",
        size="jumlah_kasus_positif_dbd",
        hover_data=["suhu_rata2_c", "kelembaban_rata2"],
        color_continuous_scale="Turbo"
    )
    fig_3d.update_layout(
        paper_bgcolor="#F5F7FB",
        font=dict(color="#1E3A8A")
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Tren Faktor Lingkungan")

    line_df = pd.DataFrame({
        "Faktor": ["Curah Hujan","Kelembaban","Suhu","Kepadatan Penduduk","Jumlah Banjir"],
        "Prediksi Kasus": [
            prediction_int * 0.6,
            prediction_int * 0.8,
            prediction_int,
            prediction_int * 1.2,
            prediction_int * 1.4
        ]
    })

    fig_line = px.line(
        line_df,
        x="Faktor",
        y="Prediksi Kasus",
        markers=True
    )
    fig_line.update_layout(
        plot_bgcolor="#F5F7FB",
        paper_bgcolor="#F5F7FB",
        font=dict(color="#1E3A8A")
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
