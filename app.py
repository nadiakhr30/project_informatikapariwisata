# app.py - Streamlit (Modern Design dengan Bootstrap Icons, Tanpa Emoji)

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
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS - MODERN DESIGN
# ========================================
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.11.1/font/bootstrap-icons.min.css">
<style>
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Header Modern */
    .modern-header {
        background: linear-gradient(135deg, #0f2b3d 0%, #1a4a6f 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .modern-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    .modern-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Card Result */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1a4a6f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    .result-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a4a6f;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .result-detail {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid #e9ecef;
    }
    .detail-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        color: #495057;
    }
    .detail-item i {
        color: #1a4a6f;
        width: 20px;
        font-size: 1rem;
    }
    .score-badge {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Sidebar Modern */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Stat Box */
    .stat-box {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a4a6f;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1a4a6f, #0f2b3d);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Info Box */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #1a4a6f;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #6c757d;
        font-size: 0.85rem;
        border-top: 1px solid #e9ecef;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODEL
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
# MAPPING KOTA KE PROVINSI
# ========================================
kota_ke_provinsi = {
    'surabaya': 'Jawa Timur', 'malang': 'Jawa Timur', 'banyuwangi': 'Jawa Timur',
    'semarang': 'Jawa Tengah', 'solo': 'Jawa Tengah', 'magelang': 'Jawa Tengah',
    'bandung': 'Jawa Barat', 'bogor': 'Jawa Barat', 'bekasi': 'Jawa Barat',
    'denpasar': 'Bali', 'kuta': 'Bali', 'ubud': 'Bali',
    'yogyakarta': 'Daerah Istimewa Yogyakarta', 'jogja': 'Daerah Istimewa Yogyakarta',
    'sleman': 'Daerah Istimewa Yogyakarta', 'bantul': 'Daerah Istimewa Yogyakarta',
    'jakarta': 'DKI Jakarta', 'tangerang': 'Banten'
}

def ekstrak_lokasi_dari_input(user_input):
    user_lower = user_input.lower()
    for kota, provinsi in kota_ke_provinsi.items():
        if kota in user_lower:
            return provinsi, kota.title()
    for prov in df_clean['provinsi'].unique():
        if prov.lower() in user_lower:
            return prov, None
    return None, None

def ekstrak_kategori_dari_input(user_input):
    user_lower = user_input.lower()
    for kat in df_clean['kategori'].unique():
        if kat in user_lower:
            return kat
    return None

def preprocess_query(teks):
    if pd.isna(teks):
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'[^\x00-\x7F]+', ' ', teks)
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def cari_wisata(kategori=None, provinsi=None, nama_wisata=None, deskripsi=None, top_n=10):
    
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

def tampilkan_hasil_modern(hasil, judul):
    if hasil is None or len(hasil) == 0:
        st.warning("Tidak ada rekomendasi yang ditemukan.")
        return
    
    st.markdown(f'<div style="margin: 1rem 0;"><span class="score-badge"><i class="bi bi-check-circle-fill"></i> {len(hasil)} Rekomendasi Ditemukan</span></div>', unsafe_allow_html=True)
    
    for i, (_, row) in enumerate(hasil.iterrows(), 1):
        skor_persen = row['skor'] * 100
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">
                <i class="bi bi-geo-alt-fill"></i> {i}. {row['nama_wisata']}
                <span class="score-badge" style="margin-left: auto;">
                    <i class="bi bi-star-fill"></i> {skor_persen:.1f}%
                </span>
            </div>
            <div class="result-detail">
                <div class="detail-item"><i class="bi bi-tag-fill"></i> <strong>Kategori:</strong> {row['kategori']}</div>
                <div class="detail-item"><i class="bi bi-building"></i> <strong>Provinsi:</strong> {row['provinsi']}</div>
                <div class="detail-item"><i class="bi bi-pin-map-fill"></i> <strong>Kota/Kab:</strong> {row['kota_kabupaten']}</div>
                <div class="detail-item"><i class="bi bi-graph-up"></i> <strong>Skor:</strong> {row['skor']:.4f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if pd.notna(row.get('alamat')):
            st.markdown(f'<div class="detail-item"><i class="bi bi-house-door"></i> <strong>Alamat:</strong> {str(row["alamat"])[:150]}...</div>', unsafe_allow_html=True)
        
        with st.expander("📖 Lihat Deskripsi Lengkap"):
            st.write(row['deskripsi_bersih'])
        
        st.markdown("</div>", unsafe_allow_html=True)

# ========================================
# HEADER MODERN
# ========================================
st.markdown("""
<div class="modern-header">
    <h1><i class="bi bi-compass"></i> SISTEM REKOMENDASI WISATA INDONESIA</h1>
    <p>Temukan destinasi wisata terbaik sesuai keinginan Anda</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR MODERN
# ========================================
with st.sidebar:
    st.markdown("## <i class='bi bi-sliders2'></i> Pengaturan", unsafe_allow_html=True)
    
    metode = st.selectbox(
        "Pilih Metode",
        ["Kategori", "Nama Wisata", "Deskripsi"],
        format_func=lambda x: f"📌 {x}"
    )
    
    top_n = st.slider("Jumlah Rekomendasi", 5, 30, 10, 5)
    
    st.markdown("---")
    st.markdown("## <i class='bi bi-bar-chart-stats'></i> Statistik", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(df_clean)}</div>
            <div class="stat-label">Total Wisata</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{df_clean['kategori'].nunique()}</div>
            <div class="stat-label">Kategori</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{df_clean['provinsi'].nunique()}</div>
            <div class="stat-label">Provinsi</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(df_clean)}</div>
            <div class="stat-label">Destinasi</div>
        </div>
        """, unsafe_allow_html=True)

# ========================================
# MAIN CONTENT - KATEGORI
# ========================================
if metode == "Kategori":
    st.markdown("## <i class='bi bi-tags'></i> Rekomendasi Berdasarkan Kategori", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        kategori = st.selectbox("Pilih Kategori", sorted(df_clean['kategori'].unique()))
    with col2:
        filter_lokasi = st.checkbox("Filter berdasarkan lokasi")
        lokasi = st.text_input("Kota/Provinsi", disabled=not filter_lokasi, placeholder="Contoh: yogyakarta, bali")
    
    if st.button("Cari Rekomendasi", use_container_width=True):
        with st.spinner("Mencari rekomendasi terbaik..."):
            prov_filter = None
            if filter_lokasi and lokasi:
                prov_filter, _ = ekstrak_lokasi_dari_input(lokasi)
            
            hasil = cari_wisata(kategori=kategori, provinsi=prov_filter, top_n=top_n)
            
            if prov_filter:
                st.markdown(f'<div class="info-box"><i class="bi bi-info-circle-fill"></i> Menampilkan {kategori} di {prov_filter}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-box"><i class="bi bi-info-circle-fill"></i> Menampilkan semua {kategori} di seluruh Indonesia</div>', unsafe_allow_html=True)
            
            tampilkan_hasil_modern(hasil, f"Rekomendasi Kategori {kategori.upper()}")

# ========================================
# MAIN CONTENT - NAMA WISATA
# ========================================
elif metode == "Nama Wisata":
    st.markdown("## <i class='bi bi-search'></i> Rekomendasi Berdasarkan Nama Wisata", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        nama_wisata = st.selectbox("Pilih Wisata Referensi", sorted(df_clean['nama_wisata'].tolist()))
    with col2:
        filter_lokasi = st.checkbox("Filter berdasarkan lokasi")
        lokasi = st.text_input("Kota/Provinsi", disabled=not filter_lokasi, placeholder="Contoh: yogyakarta, bali")
    
    if st.button("Cari Wisata Mirip", use_container_width=True):
        with st.spinner("Mencari wisata yang mirip..."):
            prov_filter = None
            if filter_lokasi and lokasi:
                prov_filter, _ = ekstrak_lokasi_dari_input(lokasi)
            
            target = df_clean[df_clean['nama_wisata'] == nama_wisata].iloc[0]
            
            st.markdown(f"""
            <div class="info-box">
                <i class="bi bi-pin-map-fill"></i> <strong>Wisata Referensi:</strong> {nama_wisata}
                <br><i class="bi bi-tag-fill"></i> Kategori: {target['kategori']} | <i class="bi bi-building"></i> Provinsi: {target['provinsi']}
            </div>
            """, unsafe_allow_html=True)
            
            hasil = cari_wisata(nama_wisata=nama_wisata, provinsi=prov_filter, top_n=top_n)
            tampilkan_hasil_modern(hasil, f"Wisata Mirip {nama_wisata}")

# ========================================
# MAIN CONTENT - DESKRIPSI
# ========================================
else:
    st.markdown("## <i class='bi bi-pencil-square'></i> Rekomendasi Berdasarkan Deskripsi", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <i class="bi bi-lightbulb-fill"></i> <strong>Contoh deskripsi:</strong><br>
        • "gunung tertinggi"<br>
        • "pantai terindah"<br>
        • "museum bersejarah"<br>
        • "air terjun di malang"
    </div>
    """, unsafe_allow_html=True)
    
    deskripsi = st.text_area("Tuliskan deskripsi Anda", height=100, placeholder="Contoh: gunung tertinggi di jawa timur")
    
    if st.button("Cari Rekomendasi", use_container_width=True):
        if deskripsi:
            with st.spinner("Menganalisis deskripsi Anda..."):
                kata_kunci = preprocess_query(deskripsi)
                kategori_terdeteksi = ekstrak_kategori_dari_input(deskripsi)
                provinsi_terdeteksi, kota_terdeteksi = ekstrak_lokasi_dari_input(deskripsi)
                
                hasil = None
                kata_sifat = ['tinggi', 'tertinggi', 'indah', 'terindah', 'bersejarah', 'populer', 'terkenal', 'bagus']
                ada_kata_sifat = any(kata in kata_kunci for kata in kata_sifat)
                
                if ada_kata_sifat:
                    st.markdown('<div class="info-box"><i class="bi bi-search-heart"></i> Mencari berdasarkan kata kunci di semua wisata...</div>', unsafe_allow_html=True)
                    if provinsi_terdeteksi:
                        st.markdown(f'<div class="info-box"><i class="bi bi-geo-alt-fill"></i> Filter lokasi: {provinsi_terdeteksi}</div>', unsafe_allow_html=True)
                        hasil = cari_wisata(deskripsi=kata_kunci, provinsi=provinsi_terdeteksi, top_n=top_n)
                    else:
                        hasil = cari_wisata(deskripsi=kata_kunci, top_n=top_n)
                    
                    tampilkan_hasil_modern(hasil, "Rekomendasi Berdasarkan Deskripsi")
                
                elif kategori_terdeteksi:
                    st.markdown(f'<div class="info-box"><i class="bi bi-tag-fill"></i> Mendeteksi kategori: {kategori_terdeteksi}</div>', unsafe_allow_html=True)
                    if provinsi_terdeteksi:
                        st.markdown(f'<div class="info-box"><i class="bi bi-geo-alt-fill"></i> Filter lokasi: {provinsi_terdeteksi}</div>', unsafe_allow_html=True)
                        hasil = cari_wisata(kategori=kategori_terdeteksi, provinsi=provinsi_terdeteksi, top_n=top_n)
                    else:
                        hasil = cari_wisata(kategori=kategori_terdeteksi, top_n=top_n)
                    
                    if hasil is not None and len(hasil) > 0:
                        judul = f"Rekomendasi {kategori_terdeteksi.upper()}"
                        if provinsi_terdeteksi:
                            judul += f" di {provinsi_terdeteksi.upper()}"
                        tampilkan_hasil_modern(hasil, judul)
                    else:
                        st.warning(f"Tidak ada {kategori_terdeteksi} yang ditemukan.")
                
                else:
                    st.markdown('<div class="info-box"><i class="bi bi-search"></i> Mencari berdasarkan deskripsi bebas...</div>', unsafe_allow_html=True)
                    if provinsi_terdeteksi:
                        st.markdown(f'<div class="info-box"><i class="bi bi-geo-alt-fill"></i> Filter lokasi: {provinsi_terdeteksi}</div>', unsafe_allow_html=True)
                        hasil = cari_wisata(deskripsi=kata_kunci, provinsi=provinsi_terdeteksi, top_n=top_n)
                    else:
                        hasil = cari_wisata(deskripsi=kata_kunci, top_n=top_n)
                    
                    tampilkan_hasil_modern(hasil, "Rekomendasi Berdasarkan Deskripsi")
        else:
            st.error("Masukkan deskripsi terlebih dahulu.")

# ========================================
# FOOTER
# ========================================
st.markdown("""
<div class="footer">
    <i class="bi bi-tree-fill"></i> Sistem Rekomendasi Wisata Indonesia 
    | Content-Based Filtering dengan TF-IDF & Cosine Similarity
    | <i class="bi bi-database"></i> Data Wisata Nusantara
</div>
""", unsafe_allow_html=True)