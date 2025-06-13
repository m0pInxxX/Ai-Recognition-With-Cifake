import os
import numpy as np
import pickle
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from utils_feature import extract_all_features

# Cek ketersediaan CUDA
if tf.test.is_built_with_cuda():
    print("CUDA tersedia. Menggunakan GPU untuk training...")
    # Aktifkan memory growth untuk menghindari alokasi memori berlebihan
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("CUDA tidak tersedia. Menggunakan CPU...")

def load_model_and_scaler(model_path, scaler_path=None):
    """Muat model dan scaler"""
    try:
        # Muat model
        model = tf.keras.models.load_model(model_path)
        
        # Muat scaler jika disediakan
        scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        print(f"Error memuat model atau scaler: {e}")
        return None, None

def visualize_pca(features, labels, n_components=2):
    """Visualisasi PCA dari fitur gambar"""
    # Standardisasi fitur
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Terapkan PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    # Plot hasil PCA
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Visualisasi PCA dari Fitur Gambar')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Tambahkan legend
    plt.legend(['Gambar Asli', 'Gambar AI'])
    
    # Plot scree plot (explained variance ratio)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.title('Scree Plot: Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return pca, features_pca

def evaluate_model_on_test_data(model, features, labels, scaler=None):
    """Evaluasi model pada data uji"""
    # Pra-proses fitur jika scaler tersedia
    if scaler:
        features = scaler.transform(features)
    
    # Prediksi
    y_pred_prob = model.predict(features)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Tampilkan hasil evaluasi
    print("\nHasil Evaluasi:")
    print(classification_report(labels, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labels, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return y_pred, y_pred_prob

def classify_single_image(model, image_path, scaler=None):
    """Klasifikasi gambar tunggal"""
    try:
        # Baca gambar
        image = imread(image_path)
        
        # Ekstrak fitur
        features = extract_all_features(image)
        
        # Reshape features untuk prediksi
        features = features.reshape(1, -1)
        
        # Pra-proses fitur jika scaler tersedia
        if scaler:
            features = scaler.transform(features)
        
        # Prediksi
        prediction_prob = model.predict(features)[0][0]
        prediction_class = 'AI' if prediction_prob > 0.5 else 'Asli'
        
        print(f"\nHasil Klasifikasi untuk {image_path}:")
        print(f"Kelas: {prediction_class}")
        print(f"Probabilitas gambar AI: {prediction_prob:.4f}")
        print(f"Probabilitas gambar asli: {1 - prediction_prob:.4f}")
        
        return prediction_class, prediction_prob
    except Exception as e:
        print(f"Error mengklasifikasi gambar {image_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluasi model untuk deteksi gambar AI")
    parser.add_argument("--model", type=str, required=True, help="Path ke model terlatih (.keras)")
    parser.add_argument("--scaler", type=str, help="Path ke scaler yang digunakan untuk pra-proses data (.pkl)")
    parser.add_argument("--test_data", type=str, help="Path ke file data uji (.pkl) yang dihasilkan oleh extract_features.py")
    parser.add_argument("--image", type=str, help="Path ke gambar tunggal untuk diklasifikasi")
    parser.add_argument("--pca_components", type=int, default=2, help="Jumlah komponen PCA untuk visualisasi (default: 2)")
    
    args = parser.parse_args()
    
    # Muat model dan scaler
    model, scaler = load_model_and_scaler(args.model, args.scaler)
    
    if model is None:
        print("Gagal memuat model. Keluar...")
        return
    
    # Jika data uji disediakan, evaluasi model pada data uji
    if args.test_data:
        try:
            with open(args.test_data, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']
            
            print(f"Data uji dimuat. Jumlah sampel: {len(labels)}")
            print(f"Jumlah gambar AI: {np.sum(labels)}")
            print(f"Jumlah gambar asli: {len(labels) - np.sum(labels)}")
            
            # Visualisasi PCA
            print("\nMembuat visualisasi PCA...")
            pca, features_pca = visualize_pca(features, labels, n_components=args.pca_components)
            
            # Evaluasi model
            print("\nMengevaluasi model...")
            y_pred, y_pred_prob = evaluate_model_on_test_data(model, features, labels, scaler)
            
        except Exception as e:
            print(f"Error memuat atau mengevaluasi data uji: {e}")
    
    # Jika gambar tunggal disediakan, klasifikasi gambar tersebut
    if args.image:
        if os.path.exists(args.image):
            pred_class, pred_prob = classify_single_image(model, args.image, scaler)
        else:
            print(f"Gambar {args.image} tidak ditemukan.")

if __name__ == "__main__":
    main() 