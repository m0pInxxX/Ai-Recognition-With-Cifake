import os
import numpy as np
import pickle
import argparse
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from skimage.io import imread
from utils_feature import extract_all_features

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

def evaluate_model_on_test_data(model, x_test, y_test, scaler=None):
    """Evaluasi model pada data uji"""
    # Pra-proses data uji jika scaler tersedia
    if scaler:
        x_test_scaled = scaler.transform(x_test)
    else:
        x_test_scaled = x_test
    
    # Prediksi
    y_pred_prob = model.predict(x_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Cetak laporan klasifikasi
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Asli', 'AI']))
    
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
    plt.savefig('confusion_matrix_evaluation.png')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('precision_recall_curve.png')
    
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