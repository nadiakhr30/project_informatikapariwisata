# app.py - Streamlit (TANPA build ulang TF-IDF)

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
    layout="wide"
)

st.title("🏝️ SISTEM REKOMENDASI WISATA INDONESIA")
st.markdown("---")

# ========================================
# LOAD MODEL YANG SUDAH DISIMPAN (CACHE)
# ========================================
@st.cache_resource
def load_model():
    # Load TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    # Load TF-IDF matrix
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    # Load data clean
    df_clean = pd.read_csv('df_clean.csv')
    
    return tfidf_vectorizer, tfidf_matrix, df_clean

# Load semua model (hanya sekali)
tfidf_vectorizer, tfidf_matrix, df_clean = load_model()

st.sidebar.success(f"✅ Model loaded! Data: {len(df_clean)} wisata")

# ========================================
# PREPROCESSING UNTUK INPUT USER (MINIMAL)
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
# FUNGSI REKOMENDASI (PAKAI MODEL YANG SUDAH LOAD)
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
st.sidebar.header("🔍 Filter Pencarian")
metode = st.sidebar.selectbox(
    "Pilih Metode:",
    ["Berdasarkan Kategori", "Berdasarkan Nama Wisata", "Berdasarkan Deskripsi"]
)
top_n = st.sidebar.slider("Jumlah Rekomendasi:", 5, 30, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistik")
st.sidebar.write(f"Total Wisata: {len(df_clean)}")
st.sidebar.write(f"Kategori: {df_clean['kategori'].nunique()}")
st.sidebar.write(f"Provinsi: {df_clean['provinsi'].nunique()}")

# ========================================
# MAIN CONTENT
# ========================================
if metode == "Berdasarkan Kategori":
    st.subheader("Rekomendasi Berdasarkan Kategori")
    
    col1, col2 = st.columns(2)
    with col1:
        kategori = st.selectbox("Pilih Kategori:", sorted(df_clean['kategori'].unique()))
    with col2:
        filter_lokasi = st.checkbox("Filter lokasi")
        lokasi = st.text_input("Kota/Provinsi:", disabled=not filter_lokasi)
    
    if st.button("Cari", type="primary"):
        with st.spinner("Mencari..."):
            prov_filter = ekstrak_lokasi(lokasi)[0] if filter_lokasi and lokasi else None
            hasil = cari_wisata(kategori=kategori, provinsi=prov_filter, top_n=top_n)
            
            if hasil is not None and len(hasil) > 0:
                for i, (_, row) in enumerate(hasil.iterrows(), 1):
                    with st.expander(f"{i}. {row['nama_wisata']}"):
                        st.write(f"**Kategori:** {row['kategori']}")
                        st.write(f"**Provinsi:** {row['provinsi']}")
                        st.write(f"**Kota/Kab:** {row['kota_kabupaten']}")
                        st.write(f"**Skor:** {row['skor']:.4f}")
            else:
                st.warning("Tidak ada rekomendasi.")

elif metode == "Berdasarkan Nama Wisata":
    st.subheader("Rekomendasi Berdasarkan Nama Wisata")
    
    col1, col2 = st.columns(2)
    with col1:
        nama_wisata = st.selectbox("Pilih Wisata:", sorted(df_clean['nama_wisata'].tolist()))
    with col2:
        filter_lokasi = st.checkbox("Filter lokasi")
        lokasi = st.text_input("Kota/Provinsi:", disabled=not filter_lokasi)
    
    if st.button("Cari Wisata Mirip", type="primary"):
        with st.spinner("Mencari..."):
            prov_filter = ekstrak_lokasi(lokasi)[0] if filter_lokasi and lokasi else None
            
            target = df_clean[df_clean['nama_wisata'] == nama_wisata].iloc[0]
            st.info(f"**Wisata referensi:** {nama_wisata} ({target['kategori']}, {target['provinsi']})")
            
            hasil = cari_wisata(nama_wisata=nama_wisata, provinsi=prov_filter, top_n=top_n)
            
            if hasil is not None and len(hasil) > 0:
                for i, (_, row) in enumerate(hasil.iterrows(), 1):
                    with st.expander(f"{i}. {row['nama_wisata']}"):
                        st.write(f"**Kategori:** {row['kategori']}")
                        st.write(f"**Provinsi:** {row['provinsi']}")
                        st.write(f"**Skor:** {row['skor']:.4f}")
            else:
                st.warning("Tidak ada wisata yang mirip.")

else:
    st.subheader("Rekomendasi Berdasarkan Deskripsi")
    
    st.markdown("Contoh: 'pantai dengan pasir putih dan ombak tenang'")
    deskripsi = st.text_area("Deskripsi:", height=100)
    
    col1, col2 = st.columns(2)
    with col1:
        filter_lokasi = st.checkbox("Filter lokasi")
        lokasi = st.text_input("Kota/Provinsi:", disabled=not filter_lokasi)
    
    if st.button("Cari", type="primary"):
        if deskripsi:
            with st.spinner("Mencari..."):
                prov_filter = ekstrak_lokasi(lokasi)[0] if filter_lokasi and lokasi else None
                hasil = cari_wisata(deskripsi=deskripsi, provinsi=prov_filter, top_n=top_n)
                
                if hasil is not None and len(hasil) > 0:
                    for i, (_, row) in enumerate(hasil.iterrows(), 1):
                        with st.expander(f"{i}. {row['nama_wisata']}"):
                            st.write(f"**Kategori:** {row['kategori']}")
                            st.write(f"**Provinsi:** {row['provinsi']}")
                            st.write(f"**Skor:** {row['skor']:.4f}")
                else:
                    st.warning("Tidak ada rekomendasi.")
        else:
            st.error("Masukkan deskripsi.")