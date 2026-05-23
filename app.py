# app.py - Streamlit (Logika sama persis dengan notebook, tetap load model .pkl)

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

# Custom CSS
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
        font-size: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏝️ SISTEM REKOMENDASI WISATA INDONESIA</h1>
    <p>Temukan destinasi wisata terbaik sesuai keinginan Anda</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODEL (CACHE)
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
# MAPPING KOTA KE PROVINSI (LENGKAP)
# ========================================
kota_ke_provinsi = {
    # Jawa Timur
    'surabaya': 'Jawa Timur', 'malang': 'Jawa Timur', 'kediri': 'Jawa Timur', 'blitar': 'Jawa Timur',
    'madiun': 'Jawa Timur', 'banyuwangi': 'Jawa Timur', 'probolinggo': 'Jawa Timur', 'pasuruan': 'Jawa Timur',
    'mojokerto': 'Jawa Timur', 'jember': 'Jawa Timur', 'situbondo': 'Jawa Timur', 'bondowoso': 'Jawa Timur',
    'lumajang': 'Jawa Timur', 'nganjuk': 'Jawa Timur', 'ponorogo': 'Jawa Timur', 'trenggalek': 'Jawa Timur',
    'tulungagung': 'Jawa Timur', 'pacitan': 'Jawa Timur', 'magetan': 'Jawa Timur', 'ngawi': 'Jawa Timur',
    'bojonegoro': 'Jawa Timur', 'tuban': 'Jawa Timur', 'lamongan': 'Jawa Timur', 'gresik': 'Jawa Timur',
    'sidoarjo': 'Jawa Timur', 'jombang': 'Jawa Timur',
    
    # Jawa Tengah
    'semarang': 'Jawa Tengah', 'solo': 'Jawa Tengah', 'surakarta': 'Jawa Tengah', 'magelang': 'Jawa Tengah',
    'pekalongan': 'Jawa Tengah', 'tegal': 'Jawa Tengah', 'cilacap': 'Jawa Tengah', 'purwokerto': 'Jawa Tengah',
    'kudus': 'Jawa Tengah', 'jepara': 'Jawa Tengah', 'demak': 'Jawa Tengah', 'kendal': 'Jawa Tengah',
    'batang': 'Jawa Tengah', 'banjarnegara': 'Jawa Tengah', 'kebumen': 'Jawa Tengah', 'purbalingga': 'Jawa Tengah',
    'banyumas': 'Jawa Tengah', 'wonosobo': 'Jawa Tengah', 'temanggung': 'Jawa Tengah', 'boyolali': 'Jawa Tengah',
    'sragen': 'Jawa Tengah', 'karanganyar': 'Jawa Tengah', 'wonogiri': 'Jawa Tengah', 'sukoharjo': 'Jawa Tengah',
    'klaten': 'Jawa Tengah', 'pati': 'Jawa Tengah', 'rembang': 'Jawa Tengah', 'blora': 'Jawa Tengah',
    
    # Jawa Barat
    'bandung': 'Jawa Barat', 'bogor': 'Jawa Barat', 'bekasi': 'Jawa Barat', 'depok': 'Jawa Barat',
    'cimahi': 'Jawa Barat', 'tasikmalaya': 'Jawa Barat', 'ciamis': 'Jawa Barat', 'banjar': 'Jawa Barat',
    'sukabumi': 'Jawa Barat', 'cirebon': 'Jawa Barat', 'indramayu': 'Jawa Barat', 'majalengka': 'Jawa Barat',
    'kuningan': 'Jawa Barat', 'garut': 'Jawa Barat', 'sumedang': 'Jawa Barat', 'purwakarta': 'Jawa Barat',
    'subang': 'Jawa Barat', 'karawang': 'Jawa Barat',
    
    # Bali
    'denpasar': 'Bali', 'badung': 'Bali', 'gianyar': 'Bali', 'tabanan': 'Bali',
    'klungkung': 'Bali', 'bangli': 'Bali', 'karangasem': 'Bali', 'buleleng': 'Bali', 
    'jembrana': 'Bali', 'kuta': 'Bali', 'seminyak': 'Bali', 'nusa dua': 'Bali', 'ubud': 'Bali',
    
    # Yogyakarta
    'yogyakarta': 'Daerah Istimewa Yogyakarta', 'jogja': 'Daerah Istimewa Yogyakarta', 
    'sleman': 'Daerah Istimewa Yogyakarta', 'bantul': 'Daerah Istimewa Yogyakarta', 
    'gunungkidul': 'Daerah Istimewa Yogyakarta', 'kulon progo': 'Daerah Istimewa Yogyakarta',
    
    # Sumatera
    'medan': 'Sumatera Utara', 'binjai': 'Sumatera Utara', 'pematangsiantar': 'Sumatera Utara',
    'palembang': 'Sumatera Selatan', 'lubuklinggau': 'Sumatera Selatan', 'pagar alam': 'Sumatera Selatan',
    'padang': 'Sumatera Barat', 'bukittinggi': 'Sumatera Barat', 'payakumbuh': 'Sumatera Barat',
    'pekanbaru': 'Riau', 'dumai': 'Riau', 'batam': 'Kepulauan Riau', 'tanjungpinang': 'Kepulauan Riau',
    'bandar lampung': 'Lampung', 'metro': 'Lampung', 'jambi': 'Jambi', 'sungai penuh': 'Jambi',
    'bengkulu': 'Bengkulu', 'pangkalpinang': 'Kepulauan Bangka Belitung',
    
    # Sulawesi
    'makassar': 'Sulawesi Selatan', 'parepare': 'Sulawesi Selatan', 'palopo': 'Sulawesi Selatan',
    'manado': 'Sulawesi Utara', 'bitung': 'Sulawesi Utara', 'tomohon': 'Sulawesi Utara', 'kotamobagu': 'Sulawesi Utara',
    'palu': 'Sulawesi Tengah', 'kendari': 'Sulawesi Tenggara', 'baubau': 'Sulawesi Tenggara',
    'gorontalo': 'Gorontalo', 'mamuju': 'Sulawesi Barat',
    
    # Kalimantan
    'balikpapan': 'Kalimantan Timur', 'samarinda': 'Kalimantan Timur', 'bontang': 'Kalimantan Timur',
    'pontianak': 'Kalimantan Barat', 'singkawang': 'Kalimantan Barat',
    'banjarmasin': 'Kalimantan Selatan', 'banjarbaru': 'Kalimantan Selatan',
    'palangkaraya': 'Kalimantan Tengah', 'tanjungselor': 'Kalimantan Utara',
    
    # Maluku & Papua
    'ambon': 'Maluku', 'ternate': 'Maluku Utara', 'tidore': 'Maluku Utara',
    'jayapura': 'Papua', 'merauke': 'Papua Selatan', 'manokwari': 'Papua Barat',
    
    # Nusa Tenggara
    'mataram': 'Nusa Tenggara Barat', 'bima': 'Nusa Tenggara Barat',
    'kupang': 'Nusa Tenggara Timur', 'ende': 'Nusa Tenggara Timur', 'maumere': 'Nusa Tenggara Timur',
    
    # Lainnya
    'jakarta': 'DKI Jakarta', 'tangerang': 'Banten', 'cilegon': 'Banten', 'serang': 'Banten',
    'aceh': 'Aceh', 'banda aceh': 'Aceh', 'lhokseumawe': 'Aceh', 'langsa': 'Aceh'
}


def ekstrak_lokasi_dari_input(user_input):
    """Ekstrak lokasi dari input user (kota atau provinsi) Returns: (provinsi, kota)"""
    user_lower = user_input.lower()
    
    for kota, provinsi in kota_ke_provinsi.items():
        if kota in user_lower:
            return provinsi, kota.title()
    
    for prov in df_clean['provinsi'].unique():
        if prov.lower() in user_lower:
            return prov, None
    
    return None, None


def preprocess_query(teks):
    """Preprocessing untuk deskripsi"""
    if pd.isna(teks):
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'[^\x00-\x7F]+', ' ', teks)
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks


def cari_wisata(kategori=None, provinsi=None, nama_wisata=None, deskripsi=None, top_n=10):
    """Rekomendasi wisata dengan Cosine Similarity - DENGAN FILTER OTOMATIS"""
    
    if sum([x is not None for x in [kategori, nama_wisata, deskripsi]]) != 1:
        return None
    
    data = df_clean.copy()
    if provinsi:
        data = data[data['provinsi'] == provinsi]
        if len(data) == 0:
            return None
    
    # METODE KATEGORI
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
    
    # METODE NAMA WISATA
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
    
    # METODE DESKRIPSI
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


def tampilkan_hasil(hasil, judul):
    """Menampilkan hasil rekomendasi di Streamlit"""
    if hasil is None or len(hasil) == 0:
        st.warning("Tidak ada rekomendasi yang ditemukan.")
        return
    
    st.success(f"✨ Menampilkan {len(hasil)} rekomendasi")
    
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
                    <tr><td><b>Skor</b></td><td>{row['skor']:.4f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 Lihat Deskripsi Lengkap"):
                st.write(row['deskripsi_bersih'])


# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.markdown("## 🎯 PENGATURAN")
    
    metode = st.selectbox(
        "📌 Pilih Metode:",
        ["🏷️ Kategori", "📍 Nama Wisata", "📝 Deskripsi"]
    )
    
    st.markdown("---")
    top_n = st.slider("📊 Jumlah Rekomendasi:", 5, 30, 10, 5)
    
    st.markdown("---")
    st.markdown("## 📊 STATISTIK")
    st.metric("Total Wisata", len(df_clean))
    st.metric("Kategori", df_clean['kategori'].nunique())
    st.metric("Provinsi", df_clean['provinsi'].nunique())

# ========================================
# MAIN CONTENT - METODE KATEGORI
# ========================================
if metode == "🏷️ Kategori":
    st.markdown("## 🏷️ REKOMENDASI BERDASARKAN KATEGORI")
    
    kategori = st.selectbox("Pilih Kategori:", sorted(df_clean['kategori'].unique()))
    
    if st.button("🔍 Cari", type="primary", use_container_width=True):
        with st.spinner("Mencari rekomendasi..."):
            hasil = cari_wisata(kategori=kategori, top_n=top_n)
            tampilkan_hasil(hasil, f"Rekomendasi Kategori {kategori.upper()}")

# ========================================
# MAIN CONTENT - METODE NAMA WISATA
# ========================================
elif metode == "📍 Nama Wisata":
    st.markdown("## 📍 REKOMENDASI BERDASARKAN NAMA WISATA")
    
    nama_wisata = st.selectbox("Pilih Wisata:", sorted(df_clean['nama_wisata'].tolist()))
    
    if st.button("🔍 Cari Wisata Mirip", type="primary", use_container_width=True):
        with st.spinner("Mencari wisata yang mirip..."):
            target = df_clean[df_clean['nama_wisata'] == nama_wisata].iloc[0]
            st.info(f"📌 **Wisata Referensi:** {nama_wisata} ({target['kategori']}, {target['provinsi']})")
            
            hasil = cari_wisata(nama_wisata=nama_wisata, top_n=top_n)
            tampilkan_hasil(hasil, f"Wisata Mirip {nama_wisata}")

# ========================================
# MAIN CONTENT - METODE DESKRIPSI
# ========================================
else:
    st.markdown("## 📝 REKOMENDASI BERDASARKAN DESKRIPSI")
    
    st.info("💡 **Contoh:** 'pantai dengan pasir putih', 'gunung tertinggi di jawa timur'")
    
    deskripsi = st.text_area("Tuliskan deskripsi Anda:", height=100)
    
    if st.button("🔍 Cari", type="primary", use_container_width=True):
        if deskripsi:
            with st.spinner("Menganalisis deskripsi Anda..."):
                hasil = cari_wisata(deskripsi=deskripsi, top_n=top_n)
                tampilkan_hasil(hasil, "Rekomendasi Berdasarkan Deskripsi")
        else:
            st.error("Masukkan deskripsi terlebih dahulu.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>Sistem Rekomendasi Wisata Indonesia | Content-Based Filtering</p>", unsafe_allow_html=True)