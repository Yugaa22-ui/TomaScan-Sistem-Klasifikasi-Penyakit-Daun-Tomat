import streamlit as st
import numpy as np
import pickle

# Load model & scaler (SVM)
try:
    model = pickle.load(open("model_svm.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("File 'model_svm.pkl' atau 'scaler.pkl' tidak ditemukan. Upload file terlebih dahulu.")
    st.stop()
except Exception as e:
    st.error(f"Kesalahan saat memuat model: {e}")
    st.stop()

st.title("TomatScan – Sistem Klasifikasi Penyakit Daun Tomat")
st.write("Masukkan nilai fitur untuk melakukan prediksi kesehatan tanaman.")

# Input fitur + keterangan
leaf_spot_size = st.number_input(
    "1. Leaf Spot Size",
    min_value=0.0,
    max_value=50.0,
    step=0.1,
    value=5.0,
    help="Ukuran rata-rata bercak pada permukaan daun tomat (semakin besar, indikasi penyakit semakin kuat)."
)

leaf_color_index = st.number_input(
    "2. Leaf Color Index",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    value=50.0,
    help="Indeks perubahan warna daun, dari hijau sehat hingga menguning atau kecokelatan."
)

temperature = st.number_input(
    "3. Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    step=0.1,
    value=25.0,
    help="Suhu lingkungan di sekitar tanaman tomat saat pengamatan."
)

humidity = st.number_input(
    "4. Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    step=0.1,
    value=60.0,
    help="Persentase kelembapan udara di lingkungan tanaman."
)

# Predict button
if st.button("Prediksi Status Tanaman"):
    input_data = np.array([[leaf_spot_size, leaf_color_index, temperature, humidity]])
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][prediction]

    label = {
        0: "Healthy (Tanaman Sehat)",
        1: "Diseased (Tanaman Terinfeksi)"
    }

    st.subheader("Hasil Prediksi")

    if prediction == 0:
        st.success(
            f"Status Tanaman: **{label[prediction]}**\n\n"
            f"Confidence: {prob:.2f}"
        )
    else:
        st.error(
            f"Status Tanaman: **{label[prediction]}**\n\n"
            f"Confidence: {prob:.2f}"
        )
