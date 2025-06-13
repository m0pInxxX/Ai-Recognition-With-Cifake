# Sistem Deteksi Gambar AI

Proyek ini bertujuan untuk mengembangkan sistem yang dapat membedakan antara gambar yang dihasilkan oleh AI dan gambar asli yang diambil oleh manusia. Sistem menggunakan pendekatan hybrid yang menggabungkan deep learning dengan ekstraksi fitur tradisional.

## Fitur

- Deteksi gambar AI vs gambar asli dengan akurasi 89%
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

## Instalasi dan Persiapan

### Persyaratan Sistem
- Python 3.8 atau lebih tinggi
- CUDA-capable GPU (opsional, direkomendasikan untuk performa lebih baik)
- 8GB RAM minimum, 16GB RAM direkomendasikan
- 2GB ruang disk untuk kode dan model terlatih

### Langkah-langkah Instalasi

1. Clone repository ini:
   ```
   git clone https://github.com/username/ai-image-detector.git
   cd ai-image-detector
   ```

2. Install dependensi:
   ```
   pip install -r requirements.txt
   ```

3. Download model pre-trained (opsional, jika tidak tersedia di repository):
   ```
   python download_models.py
   ```

## Cara Penggunaan

### 1. Menggunakan GUI (Antarmuka Grafis)

GUI adalah cara termudah untuk menggunakan sistem ini bagi pengguna non-teknis:

1. Jalankan aplikasi GUI:
   ```
   python gui_detector.py
   ```

2. Dalam mode Deteksi Tunggal:
   - Klik "Browse..." untuk memilih gambar
   - Pilih model dari dropdown (gunakan "ai_detector_hybrid_best_acc.pth" untuk hasil terbaik)
   - Pilih scaler dari dropdown (gunakan "hybrid_features.pkl" jika tersedia)
   - Klik "Deteksi Gambar" untuk melihat hasil

3. Dalam mode Batch Processing:
   - Klik "Pilih Folder" atau "Pilih Beberapa Gambar"
   - Pilih model dan scaler
   - Aktifkan opsi "Export ke CSV" jika ingin menyimpan hasil
   - Klik "Proses Batch" dan tunggu hingga selesai

### 2. Menggunakan Command Line

Untuk deteksi gambar tunggal via command line:

```
python cifake_classifier.py --image path/to/image.jpg --model ai_detector_hybrid_best_acc.pth --scaler hybrid_features.pkl
```

Untuk analisis video:

```
python cifake_classifier.py --video path/to/video.mp4 --model ai_detector_hybrid_best_acc.pth --scaler hybrid_features.pkl
```

### 3. Melatih Model Sendiri

Jika ingin melatih model dengan dataset Anda sendiri:

1. Ekstrak fitur (direkomendasikan untuk dataset besar):
   ```
   python extract_features.py --input_dir dataset_folder --output_file features.pkl --num_workers 8 --include_cnn
   ```

2. Latih model hybrid:
   ```
   python train_hybrid.py --dataset features.pkl --model_output my_model.pth
   ```

3. Evaluasi model:
   ```
   python evaluate_comprehensive.py --model my_model.pth --data dataset/hybrid_features.pkl --output_dir evaluation_results
   ```

## Hasil Evaluasi

Model hybrid terbaik mencapai:
- **Akurasi**: 89.00%
- **Precision**: 89.00% (AI) dan 88.00% (Asli)
- **Recall**: 91.00% (AI) dan 86.00% (Asli)
- **F1-Score**: 90.00% (AI) dan 87.00% (Asli)

Confusion Matrix:
```
[[72 15]
 [10 97]]
```

Dimana:
- 72 gambar asli terdeteksi benar sebagai asli
- 97 gambar AI terdeteksi benar sebagai AI
- 15 gambar asli salah terdeteksi sebagai AI
- 10 gambar AI salah terdeteksi sebagai asli

## Struktur Project

```
ai-image-detector/
├── train_hybrid.py         # Training model hybrid
├── utils_feature.py        # Fungsi ekstraksi fitur
├── extract_features.py     # Ekstraksi fitur paralel dengan multiprocessing
├── evaluate_comprehensive.py # Evaluasi model
├── gui_detector.py         # Antarmuka grafis
├── cifake_classifier.py    # Command-line classifier
├── model_architecture      # Visualisasi arsitektur model
├── ai_detector_hybrid.pth  # Model terlatih
├── ai_detector_hybrid_best_acc.pth # Model dengan akurasi terbaik
├── evaluation_results/     # Hasil evaluasi model
│   ├── actual_evaluation_results.md
│   ├── actual_confusion_matrix.png
│   ├── pca_visualization.png
│   ├── feature_importance.png
│   └── ai_detector_hybrid_learning_curves.png
├── misclassified_examples/ # Contoh gambar yang salah diklasifikasi
├── dataset/               # Dataset (jika tersedia)
│   ├── ai/                # Gambar buatan AI
│   ├── real/              # Gambar asli
│   └── hybrid_features.pkl # File fitur yang diekstrak
└── requirements.txt       # Dependensi
```

## Troubleshooting

### Masalah Memori
Jika mengalami masalah kehabisan memori saat training atau proses batch:
- Kurangi batch size di `train_hybrid.py` (cari variabel `batch_size` dan kurangi nilainya)
- Resize gambar lebih kecil di fungsi `process_images_and_extract_features()`
- Gunakan mode tanpa CNN (`include_cnn=False`) untuk mengurangi penggunaan memori

### CUDA Out of Memory
Jika mendapat error CUDA out of memory:
- Kurangi batch size
- Gunakan opsi `--cpu` untuk beralih ke CPU (lebih lambat tapi tidak memerlukan GPU)

### Ekstraksi Fitur Error
Jika terjadi error saat ekstraksi fitur:
- Pastikan gambar valid dan dapat dibaca
- Periksa format gambar (hanya RGB, RGBA, dan grayscale yang didukung)
- Ubah parameter `max_size` di fungsi `process_images_and_extract_features()` untuk gambar besar

### Deteksi Gambar AI Modern
Model mungkin kesulitan mendeteksi gambar AI generasi terbaru (Midjourney v5+, DALL-E 3, Stable Diffusion XL) yang sangat realistis, terutama gambar wajah manusia. Ini merupakan keterbatasan teknologi saat ini karena:
- Model AI generatif berkembang sangat cepat
- Gambar AI modern memiliki detail wajah dan tekstur kulit yang hampir tidak dapat dibedakan dari foto asli
- Fitur yang diekstrak mungkin tidak cukup untuk mendeteksi pola halus yang membedakan AI terbaru

Untuk kasus ini:
- Pertimbangkan untuk update dataset training dengan contoh-contoh terbaru
- Gunakan model dengan versi yang lebih baru jika tersedia
- Kombinasikan dengan metode deteksi lain untuk hasil yang lebih akurat

## Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan lanjutan:
- Implementasi data augmentation untuk meningkatkan robustness
- Penggunaan model CNN yang lebih canggih (contoh: EfficientNetV2, ViT)
- Penambahan fitur untuk deteksi deepfake video
- Implementasi model ensemble untuk meningkatkan akurasi
- Optimasi kode untuk inferensi lebih cepat
- Penambahan fitur untuk deteksi gambar AI spesifik (seperti DALL-E, Midjourney, Stable Diffusion)

## Lisensi

[Masukkan informasi lisensi di sini]

## Kontak

[Masukkan informasi kontak di sini]

## Referensi

- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks - https://arxiv.org/abs/1905.11946
- Local Binary Patterns - https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
- Gray-Level Co-occurrence Matrix - https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
- Fast Fourier Transform - https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html