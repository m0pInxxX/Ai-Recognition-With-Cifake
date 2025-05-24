import os
import numpy as np
import pickle
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import gdown
import zipfile
import kaggle
import shutil

def download_cifake_dataset(output_dir='datasets/cifake'):
    """Download CIFAKE dataset menggunakan Kaggle API"""
    print("Mengunduh dataset CIFAKE...")
    
    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download dataset dari Kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('birdsarah/cifake', path=output_dir, unzip=True)
        
        print(f"Dataset CIFAKE berhasil diunduh ke {output_dir}")
        return True
    except Exception as e:
        print(f"Error mengunduh dataset CIFAKE: {e}")
        print("Pastikan Anda telah mengatur Kaggle API dengan benar.")
        print("Lihat panduan di https://github.com/Kaggle/kaggle-api")
        return False

def download_genimage_dataset(output_dir='datasets/genimage'):
    """Download GenImage dataset"""
    print("Mengunduh dataset GenImage...")
    
    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # URL ke dataset GenImage (placeholder, perlu diganti dengan URL yang benar)
        url = 'https://drive.google.com/uc?id=1z_tu9Q34TpAjP-DEzJUDEBXmUBfOsiNj'
        
        # Download dataset
        output_zip = os.path.join(output_dir, 'genimage.zip')
        gdown.download(url, output_zip, quiet=False)
        
        # Ekstrak zip
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Hapus file zip
        os.remove(output_zip)
        
        print(f"Dataset GenImage berhasil diunduh ke {output_dir}")
        return True
    except Exception as e:
        print(f"Error mengunduh dataset GenImage: {e}")
        return False

def load_dataset(dataset_choice, base_dir='datasets'):
    """Muat dan proses dataset sesuai pilihan pengguna"""
    if dataset_choice == 'cifake':
        dataset_dir = os.path.join(base_dir, 'cifake')
        if not os.path.exists(dataset_dir):
            success = download_cifake_dataset(dataset_dir)
            if not success:
                return None, None, None, None
        
        # Muat dataset CIFAKE
        x_train = np.load(os.path.join(dataset_dir, 'train', 'x_train.npy'))
        y_train = np.load(os.path.join(dataset_dir, 'train', 'y_train.npy'))
        x_test = np.load(os.path.join(dataset_dir, 'test', 'x_test.npy'))
        y_test = np.load(os.path.join(dataset_dir, 'test', 'y_test.npy'))
        
        # Reshape gambar jika perlu
        if len(x_train.shape) == 4:  # (samples, height, width, channels)
            x_train = x_train.reshape(x_train.shape[0], -1)
        if len(x_test.shape) == 4:
            x_test = x_test.reshape(x_test.shape[0], -1)
        
        # Normalisasi data
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        return x_train, y_train, x_test, y_test
    
    elif dataset_choice == 'genimage':
        dataset_dir = os.path.join(base_dir, 'genimage')
        if not os.path.exists(dataset_dir):
            success = download_genimage_dataset(dataset_dir)
            if not success:
                return None, None, None, None
        
        # Proses dataset GenImage (disesuaikan dengan struktur yang sebenarnya)
        # Sebagai contoh, asumsikan struktur folder dengan 'real' dan 'ai' subdirektori
        from extract_features import extract_features_from_directory
        
        features_file = os.path.join(dataset_dir, 'genimage_features.pkl')
        if os.path.exists(features_file):
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']
        else:
            features, labels, _ = extract_features_from_directory(dataset_dir, features_file)
        
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        return x_train, y_train, x_test, y_test
    
    elif dataset_choice == 'custom':
        # Pengguna menyediakan path ke dataset kustom mereka
        print("Untuk dataset kustom, gunakan extract_features.py untuk mengekstrak fitur terlebih dahulu.")
        features_file = input("Masukkan path ke file fitur (.pkl) yang dihasilkan oleh extract_features.py: ")
        
        try:
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']
            
            # Split dataset
            x_train, x_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            return x_train, y_train, x_test, y_test
        except Exception as e:
            print(f"Error memuat dataset kustom: {e}")
            return None, None, None, None
    
    else:
        print(f"Dataset {dataset_choice} tidak dikenali.")
        return None, None, None, None

def build_model(input_dim):
    """Buat model neural network untuk klasifikasi"""
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate_model(x_train, y_train, x_test, y_test, model_save_path, dataset_name):
    """Latih dan evaluasi model"""
    # Pra-proses data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Simpan scaler untuk digunakan nanti
    with open(f'scaler_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Buat model
    input_dim = x_train_scaled.shape[1]
    model = build_model(input_dim)
    
    # Siapkan callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy')
    ]
    
    # Latih model
    history = model.fit(
        x_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluasi model
    y_pred_prob = model.predict(x_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\nHasil Evaluasi:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Asli', 'AI'])
    plt.yticks([0, 1], ['Asli', 'AI'])
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    
    # Tambahkan anotasi di setiap sel
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{dataset_name}.png')
    
    # Plot kurva akurasi dan loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'training_history_{dataset_name}.png')
    
    # Simpan model
    model.save(model_save_path)
    print(f"Model disimpan ke {model_save_path}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Latih model untuk deteksi gambar AI")
    parser.add_argument("--dataset", type=str, choices=['cifake', 'genimage', 'custom'], 
                        default='cifake', help="Dataset yang akan digunakan")
    parser.add_argument("--model_output", type=str, 
                        default=None, help="Path untuk menyimpan model terlatih")
    
    args = parser.parse_args()
    
    # Tentukan nama file model berdasarkan dataset
    if args.model_output is None:
        args.model_output = f'ai_detector_{args.dataset}.keras'
    
    # Muat dataset
    x_train, y_train, x_test, y_test = load_dataset(args.dataset)
    
    if x_train is None:
        print("Gagal memuat dataset. Keluar...")
        return
    
    print(f"Dataset {args.dataset} berhasil dimuat.")
    print(f"Jumlah data latih: {len(x_train)}")
    print(f"Jumlah data uji: {len(x_test)}")
    print(f"Jumlah gambar AI dalam data latih: {np.sum(y_train)}")
    print(f"Jumlah gambar asli dalam data latih: {len(y_train) - np.sum(y_train)}")
    
    # Latih dan evaluasi model
    model, history = train_and_evaluate_model(x_train, y_train, x_test, y_test, args.model_output, args.dataset)
    
    print("Pelatihan selesai!")

if __name__ == "__main__":
    main() 