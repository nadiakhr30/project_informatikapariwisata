# app.py - Streamlit (VERSI DIPERBAGUS DENGAN SEMUA FITUR)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Sistem Rekomendasi Wisata Indonesia",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan lebih baik
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #f0f0f0;
        margin-top: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .stat-box {
        background-color: #667eea;
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏝️ SISTEM REKOMENDASI WISATA INDONESIA</h1>
    <p>Temukan destinasi wisata terbaik sesuai keinginan Anda</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODEL YANG SUDAH DISIMPAN (CACHE)
# ========================================
@st.cache_resource
def load_model():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    df_clean = pd.read_csv('df_clean.csv')
    return tfidf_vectorizer, tfidf_matrix, df_clean

tfidf_vectorizer, tfidf_matrix, df_clean = load_model()

# ========================================
# PREPROCESSING UNTUK INPUT USER
# ========================================
def preprocess_query(teks):
    if pd.isna(teks):
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'[^\x00-\x7F]+', ' ', teks)
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

# ========================================
# MAPPING KOTA KE PROVINSI
# ========================================
kota_ke_provinsi = {
    'surabaya': 'Jawa Timur', 'malang': 'Jawa Timur', 'banyuwangi': 'Jawa Timur',
    'semarang': 'Jawa Tengah', 'solo': 'Jawa Tengah', 'magelang': 'Jawa Tengah',
    'bandung': 'Jawa Barat', 'bogor': 'Jawa Barat', 'bekasi': 'Jawa Barat',
    'denpasar': 'Bali', 'kuta': 'Bali', 'ubud': 'Bali',
    'yogyakarta': 'Daerah Istimewa Yogyakarta', 'jogja': 'Daerah Istimewa Yogyakarta',
    'jakarta': 'DKI Jakarta', 'tangerang': 'Banten'
}

def ekstrak_lokasi(user_input):
    user_lower = user_input.lower()
    for kota, provinsi in kota_ke_provinsi.items():
        if kota in user_lower:
            return provinsi, kota.title()
    for prov in df_clean['provinsi'].unique():
        if prov.lower() in user_lower:
            return prov, None
    return None, None

# ========================================
# FUNGSI REKOMENDASI
# ========================================
def cari_wisata(kategori=None, provinsi=None, nama_wisata=None, deskripsi=None, top_n=10):
    data = df_clean.copy()
    if provinsi:
        data = data[data['provinsi'] == provinsi]
        if len(data) == 0:
            return None
    
    # Metode KATEGORI
    if kategori:
        kategori = kategori.lower()
        data = data[data['kategori'] == kategori]
        if len(data) == 0:
            return None
        
        if len(data) <= top_n:
            hasil = data.copy()
            hasil['skor'] = 1.0
            return hasil
        
        query = tfidf_vectorizer.transform([kategori])
        idx_list = data.index.tolist()
        pos_list = [df_clean.index.get_loc(i) for i in idx_list]
        scores = cosine_similarity(query, tfidf_matrix[pos_list]).flatten()
        
        top_idx = scores.argsort()[-top_n:][::-1]
        hasil = df_clean.loc[[idx_list[i] for i in top_idx]].copy()
        hasil['skor'] = scores[top_idx]
        return hasil
    
    # Metode NAMA WISATA
    if nama_wisata:
        target = df_clean[df_clean['nama_wisata'].str.lower() == nama_wisata.lower()]
        if len(target) == 0:
            return None
        
        idx_ref = target.index[0]
        matrix_pos = df_clean.index.get_loc(idx_ref)
        
        semua_scores = cosine_similarity(tfidf_matrix[matrix_pos], tfidf_matrix).flatten()
        
        data_idx = data.index.tolist()
        data_scores = [semua_scores[df_clean.index.get_loc(i)] for i in data_idx]
        
        sorted_order = np.argsort(data_scores)[::-1]
        
        hasil_indices = []
        hasil_scores = []
        for idx in sorted_order:
            if data.iloc[idx]['nama_wisata'].lower() == nama_wisata.lower():
                continue
            if len(hasil_indices) < top_n:
                hasil_indices.append(idx)
                hasil_scores.append(data_scores[idx])
            if len(hasil_indices) >= top_n:
                break
        
        if len(hasil_indices) > 0:
            hasil = data.iloc[hasil_indices].copy()
            hasil['skor'] = hasil_scores
            return hasil
        return None
    
    # Metode DESKRIPSI
    if deskripsi:
        teks_bersih = preprocess_query(deskripsi)
        if len(teks_bersih) < 3:
            return None
        
        query = tfidf_vectorizer.transform([teks_bersih])
        idx_list = data.index.tolist()
        pos_list = [df_clean.index.get_loc(i) for i in idx_list]
        scores = cosine_similarity(query, tfidf_matrix[pos_list]).flatten()
        
        top_idx = scores.argsort()[-top_n:][::-1]
        hasil = df_clean.loc[[idx_list[i] for i in top_idx]].copy()
        hasil['skor'] = scores[top_idx]
        return hasil
    
    return None

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.markdown("## 🎯 PENGATURAN")
    
    metode = st.selectbox(
        "📌 Pilih Metode Rekomendasi:",
        ["🏷️ Berdasarkan Kategori", "📍 Berdasarkan Nama Wisata", "📝 Berdasarkan Deskripsi"]
    )
    
    st.markdown("---")
    
    top_n = st.slider("📊 Jumlah Rekomendasi:", min_value=5, max_value=30, value=10, step=5)
    
    st.markdown("---")
    
    # Statistik
    st.markdown("## 📊 STATISTIK")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Wisata", len(df_clean))
    with col2:
        st.metric("Jenis Kategori", df_clean['kategori'].nunique())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Provinsi", df_clean['provinsi'].nunique())
    with col2:
        st.metric("Kota/Kab", df_clean['kota_kabupaten'].nunique())

# ========================================
# MAIN CONTENT
# ========================================

# Pilih metode
if metode == "🏷️ Berdasarkan Kategori":
    st.markdown("## 🏷️ REKOMENDASI BERDASARKAN KATEGORI")
    st.markdown("Pilih kategori wisata yang Anda minati.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        kategori = st.selectbox("Pilih Kategori:", sorted(df_clean['kategori'].unique()))
    with col2:
        filter_lokasi = st.checkbox("📍 Filter berdasarkan lokasi")
        lokasi = st.text_input("Masukkan kota atau provinsi:", disabled=not filter_lokasi, placeholder="Contoh: surabaya, bali, bandung")
    
    if st.button("🔍 Cari Rekomendasi", type="primary", use_container_width=True):
        with st.spinner("Sedang mencari rekomendasi terbaik untuk Anda..."):
            prov_filter = ekstrak_lokasi(lokasi)[0] if filter_lokasi and lokasi else None
            hasil = cari_wisata(kategori=kategori, provinsi=prov_filter, top_n=top_n)
            
            if hasil is not None and len(hasil) > 0:
                st.success(f"✨ Menampilkan {len(hasil)} rekomendasi wisata untuk Anda")
                
                for i, (_, row) in enumerate(hasil.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                            <div class="card-title">{i}. {row['nama_wisata']}</div>
                            <table style="width: 100%;">
                                <tr><td width="30%"><b>Kategori</b></td><td>{row['kategori']}</td></tr>
                                <tr><td><b>Provinsi</b></td><td>{row['provinsi']}</td></tr>
                                <tr><td><b>Kota/Kabupaten</b></td><td>{row['kota_kabupaten']}</td></tr>
                                <tr><td><b>Alamat</b></td><td>{str(row['alamat'])[:150]}...</td></tr>
                                <tr><td><b>Skor Kecocokan</b></td><td>{row['skor']:.4f}</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📖 Lihat Deskripsi Lengkap"):
                            st.write(row['deskripsi_bersih'])
            else:
                st.warning("⚠️ Tidak ada rekomendasi yang ditemukan.")

elif metode == "📍 Berdasarkan Nama Wisata":
    st.markdown("## 📍 REKOMENDASI BERDASARKAN NAMA WISATA")
    st.markdown("Cari wisata yang mirip dengan destinasi favorit Anda.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        nama_wisata = st.selectbox("Pilih Wisata Referensi:", sorted(df_clean['nama_wisata'].tolist()))
    with col2:
        filter_lokasi = st.checkbox("📍 Filter berdasarkan lokasi")
        lokasi = st.text_input("Masukkan kota atau provinsi:", disabled=not filter_lokasi, placeholder="Contoh: surabaya, bali")
    
    if st.button("🔍 Cari Wisata Mirip", type="primary", use_container_width=True):
        with st.spinner("Sedang mencari wisata yang mirip..."):
            prov_filter = ekstrak_lokasi(lokasi)[0] if filter_lokasi and lokasi else None
            
            target = df_clean[df_clean['nama_wisata'] == nama_wisata].iloc[0]
            
            st.info(f"""
            📌 **Wisata Referensi:** {nama_wisata}
            - **Kategori:** {target['kategori']}
            - **Provinsi:** {target['provinsi']}
            - **Kota/Kab:** {target['kota_kabupaten']}
            """)
            
            hasil = cari_wisata(nama_wisata=nama_wisata, provinsi=prov_filter, top_n=top_n)
            
            if hasil is not None and len(hasil) > 0:
                st.success(f"✨ Menampilkan {len(hasil)} wisata yang mirip")
                
                for i, (_, row) in enumerate(hasil.iterrows(), 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                            <div class="card-title">{i}. {row['nama_wisata']}</div>
                            <table style="width: 100%;">
                                <tr><td width="30%"><b>Kategori</b></td><td>{row['kategori']}</td></tr>
                                <tr><td><b>Provinsi</b></td><td>{row['provinsi']}</td></tr>
                                <tr><td><b>Kota/Kabupaten</b></td><td>{row['kota_kabupaten']}</td></tr>
                                <tr><td><b>Alamat</b></td><td>{str(row['alamat'])[:150]}...</td></tr>
                                <tr><td><b>Skor Kemiripan</b></td><td>{row['skor']:.4f}</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📖 Lihat Deskripsi Lengkap"):
                            st.write(row['deskripsi_bersih'])
            else:
                st.warning("⚠️ Tidak ada wisata lain yang mirip.")

else:
    st.markdown("## 📝 REKOMENDASI BERDASARKAN DESKRIPSI")
    st.markdown("Tuliskan deskripsi wisata impian Anda, sistem akan mencari yang terbaik.")
    
    st.info("💡 **Contoh deskripsi:**\n- 'pantai dengan pasir putih dan ombak yang tenang'\n- 'gunung yang cocok untuk pendaki pemula'\n- 'candi bersejarah dengan arsitektur kuno'")
    
    deskripsi = st.text_area("📝 Tuliskan deskripsi Anda:", height=120, placeholder="Contoh: pantai dengan pasir putih dan ombak tenang cocok untuk keluarga")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        filter_lokasi = st.checkbox("📍 Filter berdasarkan lokasi")
        lokasi = st.text_input("Masukkan kota atau provinsi:", disabled=not filter_lokasi, placeholder="Contoh: surabaya, bali")
    
    if st.button("🔍 Cari Rekomendasi", type="primary", use_container_width=True):
        if deskripsi:
            with st.spinner("Menganalisis deskripsi Anda dan mencari rekomendasi terbaik..."):
                prov_filter = ekstrak_lokasi(lokasi)[0] if filter_lokasi and lokasi else None
                hasil = cari_wisata(deskripsi=deskripsi, provinsi=prov_filter, top_n=top_n)
                
                if hasil is not None and len(hasil) > 0:
                    st.success(f"✨ Menampilkan {len(hasil)} rekomendasi berdasarkan deskripsi Anda")
                    
                    for i, (_, row) in enumerate(hasil.iterrows(), 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="card">
                                <div class="card-title">{i}. {row['nama_wisata']}</div>
                                <table style="width: 100%;">
                                    <tr><td width="30%"><b>Kategori</b></td><td>{row['kategori']}</td></tr>
                                    <tr><td><b>Provinsi</b></td><td>{row['provinsi']}</td></tr>
                                    <tr><td><b>Kota/Kabupaten</b></td><td>{row['kota_kabupaten']}</td></tr>
                                    <tr><td><b>Alamat</b></td><td>{str(row['alamat'])[:150]}...</td></tr>
                                    <tr><td><b>Skor Kecocokan</b></td><td>{row['skor']:.4f}</td></tr>
                                </table>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("📖 Lihat Deskripsi Lengkap"):
                                st.write(row['deskripsi_bersih'])
                else:
                    st.warning("⚠️ Tidak ada rekomendasi yang ditemukan. Coba gunakan deskripsi yang lebih spesifik.")
        else:
            st.error("❌ Silakan masukkan deskripsi terlebih dahulu.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>© 2025 Sistem Rekomendasi Wisata Indonesia | Content-Based Filtering dengan TF-IDF & Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)