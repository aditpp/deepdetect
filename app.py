import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import io

# ====== CONFIGURASI DASAR ======
st.set_page_config(page_title="DeepDetect Dashboard", page_icon="🧠", layout="wide")

# ====== DESKRIPSI MODEL ======
with st.sidebar:
    st.title("🧩 Tentang Model")
    st.markdown("""
    **DeepDetect Model**
    - Arsitektur: CNN (Convolutional Neural Network)  
    - Tujuan: Membedakan gambar *AI Generated* vs *Real*  
    - Dataset: 10.000+ gambar (AI & Real)  
    - Epoch: 1 (demo cepat)  
    - Optimizer: Adam  
    - Loss Function: Binary Crossentropy  
    """)
    st.info("💡 Model masih bisa dilatih ulang untuk meningkatkan akurasi.")

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepdetect_model.h5")
    return model

model = load_model()

# ====== GRAFIK AKURASI ======
with st.expander("📊 Lihat Grafik Akurasi Training vs Validation"):
    try:
        history_path = "training_history.csv"
        if os.path.exists(history_path):
            df = pd.read_csv(history_path)
            fig, ax = plt.subplots()
            ax.plot(df['accuracy'], label='Train Accuracy')
            ax.plot(df['val_accuracy'], label='Validation Accuracy')
            ax.set_title('Akurasi Model')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("⚠️ File `training_history.csv` tidak ditemukan.")
    except Exception as e:
        st.error(f"Terjadi error saat menampilkan grafik: {e}")

# ====== CAROUSEL CONTOH PREDIKSI ======
with st.expander("🧠 Contoh Prediksi Otomatis"):
    sample_dir = "samples/"
    if os.path.exists(sample_dir):
        cols = st.columns(3)
        sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
        for i, img_name in enumerate(sample_images[:6]):  # tampilkan 6 contoh
            img_path = os.path.join(sample_dir, img_name)
            img = Image.open(img_path).resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            pred = model.predict(img_array)
            label = "AI Generated" if pred[0][0] > 0.5 else "Real Image"
            conf = float(pred[0][0]) if label == "AI Generated" else 1 - float(pred[0][0])
            with cols[i % 3]:
                st.image(img_path, caption=f"{label} ({conf*100:.1f}%)", use_container_width=True)
    else:
        st.warning("Folder `samples/` belum ada. Tambahkan contoh gambar di situ.")

# ====== UPLOAD GAMBAR UNTUK PREDIKSI ======
st.header("📤 Unggah Gambar Sendiri untuk Prediksi")

uploaded_file = st.file_uploader("Pilih gambar (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])
log_path = "upload_log.csv"

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diunggah", use_column_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    label = "AI Generated" if pred[0][0] > 0.5 else "Real Image"
    confidence = float(pred[0][0]) if label == "AI Generated" else 1 - float(pred[0][0])

    st.success(f"✅ Prediksi: **{label}** ({confidence*100:.2f}%)")

    # Simpan log
    log_data = pd.DataFrame([{
        "nama_file": uploaded_file.name,
        "hasil_prediksi": label,
        "confidence": confidence,
    }])
    if os.path.exists(log_path):
        log_data.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_data.to_csv(log_path, index=False)

# ====== TAMPILKAN LOG UPLOAD ======
with st.expander("📜 Log Upload & Prediksi Sebelumnya"):
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        st.dataframe(log_df)
    else:
        st.info("Belum ada log upload.")
