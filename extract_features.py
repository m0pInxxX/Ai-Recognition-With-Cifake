import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import pickle
from skimage.io import imread
from utils_feature import extract_all_features

def extract_features_from_directory(directory, output_file, include_cnn=False):
    """Ekstrak fitur dari semua gambar dalam direktori"""
    features = []
    labels = []
    file_paths = []
    
    # Cek jika ada subdirektori (kelas) atau tidak
    sub_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if sub_dirs:
        # Ekstrak fitur dari setiap kelas
        for class_dir in sub_dirs:
            class_path = os.path.join(directory, class_dir)
            
            # Tentukan label berdasarkan nama direktori
            if 'ai' in class_dir.lower() or 'fake' in class_dir.lower() or 'generated' in class_dir.lower():
                label = 1  # Label untuk gambar AI
            else:
                label = 0  # Label untuk gambar asli
            
            # Dapatkan semua file gambar
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # Ekstrak fitur dari setiap gambar
            for img_file in tqdm(image_files, desc=f"Ekstraksi fitur dari {class_dir}"):
                img_path = os.path.join(class_path, img_file)
                try:
                    # Baca gambar
                    image = imread(img_path)
                    
                    # Ekstrak fitur
                    feature_vector = extract_all_features(image, include_cnn=include_cnn)
                    
                    # Tambahkan ke list
                    features.append(feature_vector)
                    labels.append(label)
                    file_paths.append(img_path)
                except Exception as e:
                    print(f"Error mengekstrak fitur dari {img_path}: {e}")
    else:
        # Dapatkan semua file gambar
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Ekstrak fitur dari setiap gambar
        for img_file in tqdm(image_files, desc="Ekstraksi fitur"):
            img_path = os.path.join(directory, img_file)
            try:
                # Baca gambar
                image = imread(img_path)
                
                # Tentukan label berdasarkan nama file
                if 'ai' in img_file.lower() or 'fake' in img_file.lower() or 'generated' in img_file.lower():
                    label = 1  # Label untuk gambar AI
                else:
                    label = 0  # Label untuk gambar asli
                
                # Ekstrak fitur
                feature_vector = extract_all_features(image, include_cnn=include_cnn)
                
                # Tambahkan ke list
                features.append(feature_vector)
                labels.append(label)
                file_paths.append(img_path)
            except Exception as e:
                print(f"Error mengekstrak fitur dari {img_path}: {e}")
    
    # Konversi ke array numpy
    features = np.array(features)
    labels = np.array(labels)
    
    # Simpan fitur ke file
    with open(output_file, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels, 'file_paths': file_paths}, f)
    
    print(f"Fitur disimpan ke {output_file}")
    print(f"Jumlah gambar: {len(labels)}")
    print(f"Jumlah gambar AI: {np.sum(labels)}")
    print(f"Jumlah gambar asli: {len(labels) - np.sum(labels)}")
    
    return features, labels, file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstrak fitur dari dataset gambar")
    parser.add_argument("--input_dir", type=str, required=True, help="Direktori input dengan gambar")
    parser.add_argument("--output_file", type=str, default="features.pkl", help="File output untuk fitur")
    parser.add_argument("--include_cnn", action="store_true", help="Sertakan fitur CNN")
    
    args = parser.parse_args()
    
    extract_features_from_directory(args.input_dir, args.output_file, args.include_cnn) 