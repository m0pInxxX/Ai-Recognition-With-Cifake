import os
import numpy as np
import argparse
import tensorflow as tf
from skimage.io import imread
import matplotlib.pyplot as plt
from utils_feature import extract_all_features
import pickle
from tensorflow.keras.models import load_model
import cv2

def load_model_and_scaler(model_path, scaler_path=None):
    """Muat model dan scaler"""
    try:
        # Muat model
        model = load_model(model_path)
        
        # Muat scaler jika disediakan
        scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        print(f"Error memuat model atau scaler: {e}")
        return None, None

def is_hybrid_model(model):
    """Cek apakah model adalah hybrid model atau tidak"""
    # Hybrid model memiliki multiple input
    return len(model.inputs) > 1

def preprocess_image_for_hybrid(image):
    """Pra-proses gambar untuk model hybrid"""
    from skimage.transform import resize
    
    # Resize ke ukuran input EfficientNetB0
    img_resized = resize(image, (224, 224), anti_aliasing=True)
    
    # Konversi ke RGB jika grayscale
    if len(img_resized.shape) == 2:
        img_resized = np.stack([img_resized, img_resized, img_resized], axis=-1)
    elif img_resized.shape[2] == 1:
        img_resized = np.concatenate([img_resized, img_resized, img_resized], axis=-1)
    elif img_resized.shape[2] == 4:
        img_resized = img_resized[:, :, :3]  # Buang channel alpha
    
    # Pra-proses untuk EfficientNetB0
    img_processed = tf.keras.applications.efficientnet.preprocess_input(img_resized * 255.0)
    
    # Ekstrak fitur handcrafted
    features = extract_all_features(image, include_cnn=False)
    
    return np.expand_dims(img_processed, axis=0), np.expand_dims(features, axis=0)

def classify_image(image_path, model_path, scaler_path=None, output_path=None):
    """Klasifikasi gambar tunggal"""
    # Muat model dan scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    if model is None:
        print("Gagal memuat model. Keluar...")
        return
    
    try:
        # Baca gambar
        image = imread(image_path)
        
        # Cek apakah model hybrid atau bukan
        if is_hybrid_model(model):
            # Pra-proses untuk hybrid model
            img_cnn, features = preprocess_image_for_hybrid(image)
            
            # Pra-proses fitur dengan scaler
            if scaler:
                features = scaler.transform(features)
            
            # Prediksi
            prediction_prob = model.predict([img_cnn, features])[0][0]
        else:
            # Ekstrak fitur untuk model tradisional
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
        
        # Tampilkan gambar dengan hasil klasifikasi
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.title(f'Klasifikasi: {prediction_class} (AI: {prediction_prob:.4f}, Asli: {1-prediction_prob:.4f})')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path)
            print(f"Hasil klasifikasi disimpan ke {output_path}")
        else:
            plt.show()
        
        return prediction_class, prediction_prob
    except Exception as e:
        print(f"Error mengklasifikasi gambar {image_path}: {e}")
        return None, None

def classify_video(video_path, model_path, scaler_path=None, output_path=None, frame_interval=30):
    """Klasifikasi video frame by frame"""
    # Muat model dan scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    if model is None:
        print("Gagal memuat model. Keluar...")
        return
    
    try:
        # Buka video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka video {video_path}")
            return
        
        # Ambil informasi video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Siapkan writer jika output disediakan
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Proses setiap frame_interval frame
            if frame_count % frame_interval == 0:
                print(f"Memproses frame {frame_count}/{total_frames}")
                
                # Konversi BGR ke RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Cek apakah model hybrid atau bukan
                if is_hybrid_model(model):
                    # Pra-proses untuk hybrid model
                    img_cnn, features = preprocess_image_for_hybrid(rgb_frame)
                    
                    # Pra-proses fitur dengan scaler
                    if scaler:
                        features = scaler.transform(features)
                    
                    # Prediksi
                    prediction_prob = model.predict([img_cnn, features])[0][0]
                else:
                    # Ekstrak fitur untuk model tradisional
                    features = extract_all_features(rgb_frame)
                    
                    # Reshape features untuk prediksi
                    features = features.reshape(1, -1)
                    
                    # Pra-proses fitur jika scaler tersedia
                    if scaler:
                        features = scaler.transform(features)
                    
                    # Prediksi
                    prediction_prob = model.predict(features)[0][0]
                
                prediction_class = 'AI' if prediction_prob > 0.5 else 'Asli'
                results.append((frame_count, prediction_class, prediction_prob))
                
                # Tambahkan hasil klasifikasi ke frame
                label = f"{prediction_class} (AI: {prediction_prob:.4f})"
                color = (0, 0, 255) if prediction_class == 'AI' else (0, 255, 0)  # Merah untuk AI, Hijau untuk Asli
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Tulis frame ke output jika disediakan
            if output_path:
                out.write(frame)
            
            # Increment frame counter
            frame_count += 1
        
        # Tutup resource
        cap.release()
        if output_path:
            out.release()
        
        print("Klasifikasi video selesai.")
        
        # Plot hasil klasifikasi
        plt.figure(figsize=(12, 6))
        frames, classes, probs = zip(*results)
        plt.plot(frames, [p for p in probs])
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.xlabel('Frame')
        plt.ylabel('Probabilitas AI')
        plt.title('Deteksi AI sepanjang Video')
        plt.savefig('video_analysis.png')
        
        return results
    except Exception as e:
        print(f"Error mengklasifikasi video {video_path}: {e}")
        return None

def download_and_setup_model(dataset='cifake'):
    """Download dan setup model jika belum ada"""
    model_file = f'ai_detector_{dataset}.keras'
    scaler_file = f'scaler_{dataset}.pkl'
    
    if not os.path.exists(model_file):
        print(f"Model {model_file} tidak ditemukan. Perlu melatih model terlebih dahulu.")
        print("Jalankan: python train_model.py --dataset cifake")
        return None, None
    
    return model_file, scaler_file

def main():
    parser = argparse.ArgumentParser(description="Klasifikasi gambar antara AI dan asli")
    parser.add_argument("--image", type=str, help="Path ke gambar untuk diklasifikasi")
    parser.add_argument("--video", type=str, help="Path ke video untuk diklasifikasi")
    parser.add_argument("--model", type=str, help="Path ke model terlatih (.keras)")
    parser.add_argument("--scaler", type=str, help="Path ke scaler yang digunakan untuk pra-proses data (.pkl)")
    parser.add_argument("--output", type=str, help="Path untuk menyimpan hasil klasifikasi")
    parser.add_argument("--dataset", type=str, default='cifake', 
                        choices=['cifake', 'genimage', 'custom', 'hybrid'],
                        help="Dataset yang digunakan untuk model default")
    
    args = parser.parse_args()
    
    # Jika model tidak disediakan, gunakan default berdasarkan dataset
    if not args.model:
        model_file, scaler_file = download_and_setup_model(args.dataset)
        if model_file:
            args.model = model_file
            args.scaler = scaler_file
    
    # Cek parameter yang diberikan
    if not args.model:
        print("Model tidak disediakan dan tidak ditemukan model default.")
        return
    
    if not args.image and not args.video:
        print("Berikan path ke gambar (--image) atau video (--video) untuk diklasifikasi.")
        return
    
    # Klasifikasi gambar
    if args.image:
        if not os.path.exists(args.image):
            print(f"Gambar {args.image} tidak ditemukan.")
            return
        
        classify_image(args.image, args.model, args.scaler, args.output)
    
    # Klasifikasi video
    if args.video:
        if not os.path.exists(args.video):
            print(f"Video {args.video} tidak ditemukan.")
            return
        
        classify_video(args.video, args.model, args.scaler, args.output)

if __name__ == "__main__":
    main() 