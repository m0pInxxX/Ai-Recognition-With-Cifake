# Sistem Deteksi Gambar AI

Proyek ini bertujuan untuk mengembangkan sistem yang dapat membedakan antara gambar yang dihasilkan oleh AI dan gambar asli yang diambil oleh manusia. Sistem menggunakan pendekatan hybrid yang menggabungkan deep learning dengan ekstraksi fitur tradisional.

## Fitur

- Deteksi gambar AI vs gambar asli dengan akurasi tinggi
- Model hybrid yang mengkombinasikan CNN (EfficientNetB0) dengan fitur handcrafted
- Ekstraksi fitur yang mencakup analisis tekstur, noise, dan domain frekuensi
- Dukungan untuk pemrosesan batch maupun deteksi gambar tunggal
- Antarmuka grafis (GUI) user-friendly untuk pengguna non-teknis
- Kemampuan untuk menganalisis video frame-by-frame
- Ekspor hasil ke format CSV untuk analisis lebih lanjut

## Cara Kerja Sistem

Sistem deteksi gambar AI ini menggunakan pendekatan hybrid yang mengkombinasikan Convolutional Neural Network (EfficientNetB0) dengan fitur-fitur handcrafted tradisional. Berikut penjelasan rinci tentang cara kerja sistem:

### 1. Arsitektur Model Hybrid

#### A. Komponen CNN
- **Backbone CNN**: Menggunakan EfficientNetB0 pre-trained pada ImageNet
- **Global Average Pooling**: Mengkonversi feature maps menjadi vektor fitur
- **Transfer Learning**: 100 layer awal EfficientNetB0 dibekukan, sisanya fine-tuning

#### B. Ekstraksi Fitur Handcrafted
- **Fitur Noise**: Estimasi noise menggunakan `estimate_sigma` dari skimage
- **Fitur Tekstur**: 
  - Local Binary Patterns (LBP) dengan metode 'uniform'
  - Gray-Level Co-occurrence Matrix (GLCM) yang mengekstrak contrast, dissimilarity, homogeneity, energy, dan correlation
- **Fitur Frekuensi**: 
  - Fast Fourier Transform (FFT) untuk menganalisis distribusi frekuensi
  - Ekstraksi statistik dari spektrum magnitudo
  - Analisis energi pada region frekuensi rendah, menengah, dan tinggi

#### C. Integrator Model
- **Concatenation**: Menggabungkan fitur CNN dan handcrafted
- **Fully-Connected Layers**:
  - Layer 1: 256 neuron dengan ReLU + BatchNorm + Dropout(0.4)
  - Layer 2: 128 neuron dengan ReLU + BatchNorm + Dropout(0.3)
  - Output Layer: 1 neuron dengan Sigmoid activation

### 2. Proses Deteksi

#### Alur Pemrosesan Gambar
1. **Pra-pemrosesan**:
   - Resize gambar ke 224x224 pixel
   - Konversi RGBA ke RGB jika diperlukan
   - Normalisasi dengan EfficientNet preprocess_input

2. **Ekstraksi Fitur Parallel**:
   - **Path CNN**: Memproses gambar melalui EfficientNetB0
   - **Path Fitur Handcrafted**: Ekstraksi fitur LBP, noise, GLCM, dan frekuensi
   - **Normalisasi Fitur**: StandardScaler digunakan untuk normalisasi fitur handcrafted

3. **Klasifikasi**:
   - Model hybrid memproses kedua input
   - Output berupa probabilitas gambar adalah buatan AI
   - Threshold 0.5: >0.5 klasifikasi sebagai AI, ≤0.5 sebagai Asli

## Penggunaan

1. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

2. Untuk melatih model hybrid:
   ```
   python train_hybrid.py --dataset dataset/path --model model_hybrid.keras
   ```

3. Untuk mengevaluasi model:
   ```
   python evaluate_model.py --model model_hybrid.keras --test_data test/dataset/path
   ```

4. Untuk klasifikasi gambar melalui command line:
   ```
   python cifake_classifier.py --image path/to/image.jpg --model model_hybrid.keras --scaler scaler_hybrid.pkl
   ```

5. Untuk menggunakan antarmuka grafis (GUI):
   ```
   python gui_detector.py
   ```

6. Untuk menganalisis video:
   ```
   python cifake_classifier.py --video path/to/video.mp4 --model model_hybrid.keras --scaler scaler_hybrid.pkl
   ```

## Antarmuka Grafis (GUI)

Aplikasi ini menyediakan GUI user-friendly dengan dua mode utama:

### Mode Deteksi Tunggal
- Pilih gambar dari komputer
- Pilih model dan scaler yang akan digunakan
- Lihat hasil deteksi dan probabilitas
- Visualisasi gambar dengan label hasil deteksi

### Mode Batch Processing
- Proses banyak gambar sekaligus dari folder atau beberapa file
- Lihat hasil dalam format tabel
- Progress bar untuk memantau proses
- Export hasil ke file CSV

## Teknologi dan Dependensi

- TensorFlow 2.8+ untuk deep learning
- scikit-image untuk ekstraksi fitur handcrafted
- scikit-learn untuk preprocessing dan evaluasi
- OpenCV untuk pemrosesan gambar dan video
- Tkinter untuk antarmuka grafis
- Matplotlib untuk visualisasi hasil

## Detail Implementasi

### Ekstraksi Fitur
- **LBP Features**: Mengekstrak pola tekstur lokal menggunakan local_binary_pattern dengan P=8, R=1
- **Noise Features**: Menganalisis karakteristik noise dengan estimate_sigma
- **GLCM Features**: Mengekstrak properti contrast, dissimilarity, homogeneity, energy, dan correlation pada berbagai jarak dan sudut
- **Frequency Features**: Menganalisis spektrum FFT dengan fokus pada distribusi energi di berbagai band frekuensi

### Pelatihan Model
- Optimasi dengan Adam optimizer
- Loss function: Binary Cross-Entropy
- Callbacks: Early Stopping, ReduceLROnPlateau, dan ModelCheckpoint
- Data augmentation: tidak diimplementasikan dalam versi ini

## Pembacaan Hasil

Hasil deteksi diberikan dalam format:
- **Status**: "AI 🤖" atau "Asli 📷"
- **Probabilitas AI**: Persentase keyakinan model bahwa gambar dibuat oleh AI
- **Probabilitas Asli**: Persentase keyakinan model bahwa gambar adalah asli

## Pengembangan Lebih Lanjut

- Implementasi data augmentation yang lebih komprehensif
- Penggunaan model CNN yang lebih canggih (contoh: EfficientNetV2, ViT)
- Penambahan fitur khusus untuk deteksi deepfake
- Analisis perbandingan berbagai arsitektur model

## Struktur Proyek

- `cifake_classifier.py`: File utama untuk klasifikasi gambar dan video
- `utils_feature.py`: Utilitas untuk ekstraksi fitur handcrafted
- `extract_features.py`: Kode untuk mengekstrak fitur dari dataset
- `train_hybrid.py`: Kode untuk melatih model hybrid
- `evaluate_model.py`: Kode untuk mengevaluasi performa model
- `gui_detector.py`: Antarmuka grafis untuk deteksi gambar
- `model_hybrid.keras`: Model terlatih yang siap digunakan
- `scaler_hybrid.pkl`: Standard scaler untuk normalisasi fitur handcrafted 

## Kesimpulan
Sistem deteksi gambar AI ini menggunakan pendekatan yang efektif yang menggabungkan deep learning dengan analisis tradisional untuk mendeteksi gambar buatan AI. Keunggulan pendekatan hybrid ini adalah kemampuannya menangkap baik fitur visual kompleks (melalui CNN) maupun artefak teknis halus dari proses generasi AI (melalui fitur handcrafted).
Sistem ini bekerja dengan menganalisis kombinasi dari tekstur, pola noise, dan karakteristik frekuensi yang sering berbeda antara gambar AI dan gambar asli. Tidak hanya melihat konsistensi garis dan tekstur, tetapi juga distribusi noise, perbedaan karakteristik statistik, dan pola frekuensi yang khas dari gambar buatan AI.