import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gaussian
from skimage.restoration import estimate_sigma
from skimage.color import rgb2gray
from skimage.transform import resize
import tensorflow as tf

def extract_lbp_features(image, P=8, R=1):
    """Ekstraksi fitur Local Binary Pattern (LBP)"""
    try:
        # Konversi ke grayscale jika gambar berwarna
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image

        # Terapkan LBP
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        
        # Hitung histogram
        n_bins = P * (P - 1) + 3
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    except Exception as e:
        print(f"Error dalam ekstraksi LBP: {e}")
        return np.zeros(P * (P - 1) + 3)

def extract_noise_features(image, patch_size=5):
    """Ekstraksi fitur tingkat noise"""
    try:
        # Konversi ke grayscale jika gambar berwarna
        if len(image.shape) == 3:
            # Estimasi noise untuk setiap channel dan ambil rata-rata
            sigma_r = estimate_sigma(image[:, :, 0])
            sigma_g = estimate_sigma(image[:, :, 1])
            sigma_b = estimate_sigma(image[:, :, 2])
            sigma = (sigma_r + sigma_g + sigma_b) / 3.0
        else:
            sigma = estimate_sigma(image)
        
        # Buat vektor fitur
        noise_features = np.array([sigma])
        
        return noise_features
    except Exception as e:
        print(f"Error dalam ekstraksi noise: {e}")
        return np.array([0.0])

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Ekstraksi fitur Gray-Level Co-occurrence Matrix (GLCM)"""
    try:
        # Konversi ke grayscale jika gambar berwarna
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
            
        # Skala ke 0-255 dan konversi ke uint8
        gray = (gray * 255).astype(np.uint8)
        
        # Kurangi jumlah level abu-abu untuk mempercepat komputasi
        gray = (gray // 32).astype(np.uint8)
        
        # Hitung GLCM
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=8, symmetric=True, normed=True)
        
        # Ekstrak properti GLCM
        contrast = graycoprops(glcm, 'contrast').ravel()
        dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
        homogeneity = graycoprops(glcm, 'homogeneity').ravel()
        energy = graycoprops(glcm, 'energy').ravel()
        correlation = graycoprops(glcm, 'correlation').ravel()
        
        # Gabungkan semua properti menjadi satu vektor fitur
        glcm_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
        
        return glcm_features
    except Exception as e:
        print(f"Error dalam ekstraksi GLCM: {e}")
        return np.zeros(len(distances) * len(angles) * 5)

def extract_frequency_features(image):
    """Ekstraksi fitur dari domain frekuensi menggunakan FFT"""
    try:
        # Konversi ke grayscale jika gambar berwarna
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # Terapkan FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Hitung statistik dari spektrum magnitudo
        mean = np.mean(magnitude_spectrum)
        std = np.std(magnitude_spectrum)
        max_val = np.max(magnitude_spectrum)
        min_val = np.min(magnitude_spectrum)
        
        # Bagi spektrum menjadi region dan hitung energi
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Hitung energi untuk region rendah, menengah, dan tinggi
        radius1 = min(center_h, center_w) // 3
        radius2 = 2 * min(center_h, center_w) // 3
        
        y, x = np.ogrid[:h, :w]
        mask_low = ((y - center_h)**2 + (x - center_w)**2 <= radius1**2)
        mask_mid = ((y - center_h)**2 + (x - center_w)**2 <= radius2**2) & ~mask_low
        mask_high = ~((y - center_h)**2 + (x - center_w)**2 <= radius2**2)
        
        energy_low = np.sum(magnitude_spectrum[mask_low]) / np.sum(mask_low)
        energy_mid = np.sum(magnitude_spectrum[mask_mid]) / np.sum(mask_mid)
        energy_high = np.sum(magnitude_spectrum[mask_high]) / np.sum(mask_high)
        
        ratio_low_high = energy_low / (energy_high + 1e-10)
        ratio_mid_high = energy_mid / (energy_high + 1e-10)
        
        freq_features = np.array([mean, std, max_val, min_val, 
                                energy_low, energy_mid, energy_high,
                                ratio_low_high, ratio_mid_high])
        
        return freq_features
    except Exception as e:
        print(f"Error dalam ekstraksi fitur frekuensi: {e}")
        return np.zeros(9)

def extract_cnn_features(image, model=None):
    """Ekstraksi fitur menggunakan model CNN yang sudah dilatih"""
    try:
        if model is None:
            # Gunakan EfficientNetB0 sebagai ekstraksi fitur
            base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
            model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
        
        # Pra-proses gambar
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)  # Konversi ke RGB
        elif image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=-1)  # Konversi ke RGB
        elif image.shape[2] == 4:
            image = image[:, :, :3]  # Buang channel alpha
        
        # Resize ke ukuran input model
        image = resize(image, (224, 224), anti_aliasing=True)
        
        # Pra-proses untuk EfficientNet
        image = tf.keras.applications.efficientnet.preprocess_input(image * 255.0)
        
        # Ekstraksi fitur
        features = model.predict(np.expand_dims(image, axis=0))
        
        # Flatten fitur
        features = features.flatten()
        
        # Kurangi dimensi jika terlalu besar
        if len(features) > 1024:
            # Pilih 1024 fitur pertama atau sub-sampel
            features = features[:1024]
        
        return features
    except Exception as e:
        print(f"Error dalam ekstraksi fitur CNN: {e}")
        return np.zeros(1024)

def combine_features(feature_list):
    """Menggabungkan beberapa vektor fitur menjadi satu"""
    try:
        # Pastikan semua fitur adalah array 1D
        flat_features = [f.flatten() for f in feature_list if f is not None]
        
        # Gabungkan semua fitur
        if flat_features:
            combined = np.concatenate(flat_features)
            return combined
        else:
            return np.array([])
    except Exception as e:
        print(f"Error dalam penggabungan fitur: {e}")
        return np.array([])

def extract_all_features(image, include_cnn=False, cnn_model=None):
    """Ekstrak semua fitur dari gambar"""
    try:
        # Resizing untuk konsistensi
        if image.shape[0] > 512 or image.shape[1] > 512:
            image = resize(image, (512, 512), anti_aliasing=True)
            
        # Ekstrak semua fitur
        lbp_feat = extract_lbp_features(image)
        noise_feat = extract_noise_features(image)
        glcm_feat = extract_glcm_features(image)
        freq_feat = extract_frequency_features(image)
        
        feature_list = [lbp_feat, noise_feat, glcm_feat, freq_feat]
        
        if include_cnn:
            cnn_feat = extract_cnn_features(image, cnn_model)
            feature_list.append(cnn_feat)
        
        # Gabungkan semua fitur
        all_features = combine_features(feature_list)
        
        return all_features
    except Exception as e:
        print(f"Error dalam ekstraksi semua fitur: {e}")
        if include_cnn:
            return np.zeros(1024 + 59)  # Approximate size for all features
        else:
            return np.zeros(59)  # Approximate size for non-CNN features 