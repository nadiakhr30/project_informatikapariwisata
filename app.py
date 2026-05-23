# app.py - Streamlit (LOGIKA: KATEGORI = FILTER, SIFAT = COSINE SIMILARITY)

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

st.sidebar.success(f"✅ Model loaded! Data: {len(df_clean)} wisata")

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

def tampilkan_hasil(hasil, judul):
    if hasil is None or len(hasil) == 0:
        st.warning("Tidak ada rekomendasi yang ditemukan.")
        return
    
    st.success(f"Menampilkan {len(hasil)} rekomendasi")
    
    for i, (_, row) in enumerate(hasil.iterrows(), 1):
        with st.expander(f"{i}. {row['nama_wisata']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Kategori:** {row['kategori']}")
                st.write(f"**Provinsi:** {row['provinsi']}")
                st.write(f"**Kota/Kab:** {row['kota_kabupaten']}")
            with col2:
                st.write(f"**Skor:** {row['skor']:.4f}")
            if pd.notna(row.get('alamat')):
                st.write(f"**Alamat:** {row['alamat'][:200]}...")
            with st.expander("📖 Deskripsi Lengkap"):
                st.write(row['deskripsi_bersih'])

# ========================================
# SIDEBAR
# ========================================
with st.sidebar:
    st.markdown("## 🎯 PENGATURAN")
    
    metode = st.selectbox(
        "Pilih Metode:",
        ["Kategori", "Nama Wisata", "Deskripsi"]
    )
    
    top_n = st.slider("Jumlah Rekomendasi:", 5, 30, 10, 5)
    
    st.markdown("---")
    st.markdown("## 📊 STATISTIK")
    st.metric("Total Wisata", len(df_clean))
    st.metric("Kategori", df_clean['kategori'].nunique())
    st.metric("Provinsi", df_clean['provinsi'].nunique())

# ========================================
# MAIN CONTENT
# ========================================

if metode == "Kategori":
    st.markdown("## REKOMENDASI BERDASARKAN KATEGORI")
    
    kategori = st.selectbox("Pilih Kategori:", sorted(df_clean['kategori'].unique()))
    
    col1, col2 = st.columns(2)
    with col1:
        filter_lokasi = st.checkbox("Filter berdasarkan lokasi")
    with col2:
        lokasi = st.text_input("Kota/Provinsi:", disabled=not filter_lokasi, placeholder="Contoh: yogyakarta, bali")
    
    if st.button("Cari", type="primary", use_container_width=True):
        with st.spinner("Mencari rekomendasi..."):
            prov_filter = None
            if filter_lokasi and lokasi:
                prov_filter, _ = ekstrak_lokasi_dari_input(lokasi)
            
            hasil = cari_wisata(kategori=kategori, provinsi=prov_filter, top_n=top_n)
            
            if prov_filter:
                st.info(f"📍 Menampilkan {kategori} di {prov_filter}")
            else:
                st.info(f"📍 Menampilkan semua {kategori} di seluruh Indonesia")
            
            tampilkan_hasil(hasil, f"Rekomendasi Kategori {kategori.upper()}")

elif metode == "Nama Wisata":
    st.markdown("## REKOMENDASI BERDASARKAN NAMA WISATA")
    
    nama_wisata = st.selectbox("Pilih Wisata:", sorted(df_clean['nama_wisata'].tolist()))
    
    col1, col2 = st.columns(2)
    with col1:
        filter_lokasi = st.checkbox("Filter berdasarkan lokasi")
    with col2:
        lokasi = st.text_input("Kota/Provinsi:", disabled=not filter_lokasi, placeholder="Contoh: yogyakarta, bali")
    
    if st.button("Cari Wisata Mirip", type="primary", use_container_width=True):
        with st.spinner("Mencari wisata yang mirip..."):
            prov_filter = None
            if filter_lokasi and lokasi:
                prov_filter, _ = ekstrak_lokasi_dari_input(lokasi)
            
            target = df_clean[df_clean['nama_wisata'] == nama_wisata].iloc[0]
            st.info(f"**Wisata Referensi:** {nama_wisata} ({target['kategori']}, {target['provinsi']})")
            
            if prov_filter:
                st.info(f"📍 Filter lokasi: {prov_filter}")
            
            hasil = cari_wisata(nama_wisata=nama_wisata, provinsi=prov_filter, top_n=top_n)
            tampilkan_hasil(hasil, f"Wisata Mirip {nama_wisata}")

else:
    st.markdown("## REKOMENDASI BERDASARKAN DESKRIPSI")
    
    st.info("💡 **Contoh:** 'gunung tertinggi', 'pantai terindah', 'museum bersejarah', 'air terjun di malang'")
    
    deskripsi = st.text_area("Tuliskan deskripsi Anda:", height=100)
    
    if st.button("Cari", type="primary", use_container_width=True):
        if deskripsi:
            with st.spinner("Menganalisis deskripsi Anda..."):
                
                # EKSTRAK INFORMASI
                kata_kunci = preprocess_query(deskripsi)
                kategori_terdeteksi = ekstrak_kategori_dari_input(deskripsi)
                provinsi_terdeteksi, kota_terdeteksi = ekstrak_lokasi_dari_input(deskripsi)
                
                hasil = None
                
                # LOGIKA:
                # 1. Jika ada kata sifat (tinggi, indah, bersejarah, dll) -> cari di SEMUA WISATA
                # 2. Jika hanya kategori -> cari spesifik kategori
                # 3. Jika ada lokasi -> filter berdasarkan lokasi
                
                kata_sifat = ['tinggi', 'tertinggi', 'indah', 'terindah', 'bersejarah', 'populer', 'terkenal', 'bagus']
                ada_kata_sifat = any(kata in kata_kunci for kata in kata_sifat)
                
                if ada_kata_sifat:
                    # Cari di SEMUA WISATA berdasarkan deskripsi
                    st.info("📝 Mencari berdasarkan kata kunci di semua wisata...")
                    if provinsi_terdeteksi:
                        st.info(f"📍 Filter lokasi: {provinsi_terdeteksi}")
                        hasil = cari_wisata(deskripsi=kata_kunci, provinsi=provinsi_terdeteksi, top_n=top_n)
                    else:
                        hasil = cari_wisata(deskripsi=kata_kunci, top_n=top_n)
                    
                    tampilkan_hasil(hasil, "Rekomendasi Berdasarkan Deskripsi")
                
                elif kategori_terdeteksi:
                    # Cari berdasarkan KATEGORI
                    st.info(f"🏷️ Mendeteksi kategori: {kategori_terdeteksi}")
                    if provinsi_terdeteksi:
                        st.info(f"📍 Filter lokasi: {provinsi_terdeteksi}")
                        hasil = cari_wisata(kategori=kategori_terdeteksi, provinsi=provinsi_terdeteksi, top_n=top_n)
                    else:
                        hasil = cari_wisata(kategori=kategori_terdeteksi, top_n=top_n)
                    
                    if hasil is not None and len(hasil) > 0:
                        judul = f"Rekomendasi {kategori_terdeteksi.upper()}"
                        if provinsi_terdeteksi:
                            judul += f" di {provinsi_terdeteksi.upper()}"
                        tampilkan_hasil(hasil, judul)
                    else:
                        st.warning(f"Tidak ada {kategori_terdeteksi} yang ditemukan.")
                
                else:
                    # Cari berdasarkan DESKRIPSI BEBAS
                    st.info("📝 Mencari berdasarkan deskripsi bebas...")
                    if provinsi_terdeteksi:
                        st.info(f"📍 Filter lokasi: {provinsi_terdeteksi}")
                        hasil = cari_wisata(deskripsi=kata_kunci, provinsi=provinsi_terdeteksi, top_n=top_n)
                    else:
                        hasil = cari_wisata(deskripsi=kata_kunci, top_n=top_n)
                    
                    tampilkan_hasil(hasil, "Rekomendasi Berdasarkan Deskripsi")
                
        else:
            st.error("Masukkan deskripsi terlebih dahulu.")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#888;'>Sistem Rekomendasi Wisata Indonesia | Content-Based Filtering</p>", unsafe_allow_html=True)