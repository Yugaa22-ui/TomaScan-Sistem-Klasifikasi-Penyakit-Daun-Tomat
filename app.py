import streamlit as st
import numpy as np
import pickle

# 1. LOAD MODEL DAN SCALER
try:
    model = pickle.load(open("model_knn.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("File 'model_knn.pkl' atau 'scaler.pkl' tidak ditemukan. Pastikan file tersebut sudah di-upload.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# Judul Aplikasi
st.title("TomaScan – Sistem Klasifikasi Penyakit Daun Tomat")
st.write("Masukkan nilai fitur numerik untuk memprediksi kondisi kesehatan tanaman.")

# 2. FORM INPUT (4 FITUR NUMERIK)
leaf_spot_size = st.number_input(
    "1. Leaf Spot Size",
    min_value=0.0, max_value=50.0, step=0.1, value=5.0,
    help="Ukuran bercak daun (semakin besar semakin berisiko)."
)

leaf_color_index = st.number_input(
    "2. Leaf Color Index",
    min_value=0.0, max_value=100.0, step=0.1, value=50.0,
    help="Indeks warna daun (berkaitan dengan klorofil)."
)

temperature = st.number_input(
    "3. Temperature (°C)",
    min_value=0.0, max_value=50.0, step=0.1, value=25.0,
    help="Suhu lingkungan tanaman."
)

humidity = st.number_input(
    "4. Humidity (%)",
    min_value=0.0, max_value=100.0, step=0.1, value=60.0,
    help="Kelembaban udara sekitar tanaman."
)

# Tombol Prediksi
if st.button("Prediksi Status Tanaman"):

    # Buat array fitur (urutan HARUS sama dengan model)
    input_data = np.array([[leaf_spot_size, leaf_color_index, temperature, humidity]])

    # Scaling
    scaled_input = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(scaled_input)[0]

    # Label hasil prediksi
    disease_label = {
        0: "Healthy (Tanaman Sehat)",
        1: "Diseased (Tanaman Terinfeksi)"
    }

    result = disease_label[prediction]

    # 3. OUTPUT HASIL
    st.subheader("Hasil Prediksi")

    if prediction == 0:
        st.balloons()
        st.success(f"Status Tanaman: **{result}**")
    else:
        st.error(f"Status Tanaman: **{result}**")
