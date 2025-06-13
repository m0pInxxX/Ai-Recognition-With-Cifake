import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import os

def load_data(data_path):
    """Muat data dari file pickle"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data.get('features')
    labels = data.get('labels')
    file_paths = data.get('file_paths', [])
    
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Total sampel: {len(labels)}")
    
    return features, labels, file_paths

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Hitung metrik evaluasi model"""
    # Klasifikasi report
    report = classification_report(y_true, y_pred, target_names=['Asli', 'AI'], output_dict=True)
    print(classification_report(y_true, y_pred, target_names=['Asli', 'AI']))
    
    # Metrik
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Akurasi: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC dan AUC (jika probabilitas tersedia)
    roc_auc = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        'report': report,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def plot_confusion_matrix(cm, output_path="confusion_matrix.png"):
    """Plot dan simpan confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Asli', 'AI'], yticklabels=['Asli', 'AI'])
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix - Hybrid Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix disimpan ke {output_path}")
    
    return cm

def perform_pca(features, labels, n_components=2, output_path="pca_visualization.png"):
    """Lakukan PCA dan visualisasikan hasil"""
    # Standardisasi fitur
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    # Persentase variance yang dijelaskan
    explained_variance = pca.explained_variance_ratio_
    print(f"Variance yang dijelaskan oleh {n_components} komponen: {sum(explained_variance):.2f}")
    print(f"Variance per komponen: {explained_variance}")
    
    # Plot
    plt.figure(figsize=(12, 10))
    colors = ['blue', 'red']
    targets = [0, 1]
    labels_text = ['Asli', 'AI']
    
    for target, color, label_text in zip(targets, colors, labels_text):
        indices = labels == target
        plt.scatter(features_pca[indices, 0], features_pca[indices, 1], 
                   c=color, s=50, alpha=0.7, label=label_text)
    
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%})')
    plt.title('PCA Visualization of Image Features')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA visualization disimpan ke {output_path}")
    
    return features_pca, explained_variance

def save_evaluation_results(results, output_path="evaluation_results.md"):
    """Simpan hasil evaluasi ke file markdown"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Hasil Evaluasi Model Hybrid AI Detector\n\n")
        
        # Metriks
        f.write("## Metrik Utama\n\n")
        f.write(f"| Metrik | Nilai |\n")
        f.write(f"|--------|------|\n")
        f.write(f"| Akurasi | {results['accuracy']:.4f} |\n")
        f.write(f"| Precision | {results['precision']:.4f} |\n")
        f.write(f"| Recall | {results['recall']:.4f} |\n")
        f.write(f"| F1 Score | {results['f1_score']:.4f} |\n")
        if results['roc_auc'] is not None:
            f.write(f"| ROC AUC | {results['roc_auc']:.4f} |\n")
        f.write("\n")
        
        # Classification Report
        f.write("## Classification Report\n\n")
        f.write(f"| Class | Precision | Recall | F1-Score | Support |\n")
        f.write(f"|-------|-----------|--------|----------|--------|\n")
        
        for class_name, metrics in results['report'].items():
            if class_name in ['Asli', 'AI']:
                f.write(f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']} |\n")
        
        f.write(f"| **accuracy** | | | {results['report']['accuracy']:.4f} | {results['report']['macro avg']['support']} |\n")
        f.write(f"| **macro avg** | {results['report']['macro avg']['precision']:.4f} | {results['report']['macro avg']['recall']:.4f} | {results['report']['macro avg']['f1-score']:.4f} | {results['report']['macro avg']['support']} |\n")
        f.write(f"| **weighted avg** | {results['report']['weighted avg']['precision']:.4f} | {results['report']['weighted avg']['recall']:.4f} | {results['report']['weighted avg']['f1-score']:.4f} | {results['report']['weighted avg']['support']} |\n\n")
        
        # Confusion Matrix (text representation)
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"{results['confusion_matrix']}\n")
        f.write("```\n\n")
        
        f.write("## Visualisasi\n\n")
        f.write("Visualisasi berikut telah disimpan sebagai file gambar:\n\n")
        f.write("1. `confusion_matrix.png` - Confusion Matrix\n")
        f.write("2. `pca_visualization.png` - PCA Visualization\n")
    
    print(f"Hasil evaluasi disimpan ke {output_path}")

def main(data_path, labels_pred=None, labels_prob=None, output_dir="."):
    """Fungsi utama untuk evaluasi sederhana menggunakan file hybrid features"""
    # Pastikan output directory ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Muat data
    features, labels, file_paths = load_data(data_path)
    
    # Jika prediksi tidak disediakan, gunakan label asli sebagai "prediksi sempurna" untuk analisis data
    if labels_pred is None:
        print("Prediksi tidak disediakan, menggunakan label asli (model sempurna)")
        labels_pred = labels
    
    # Jika probabilitas tidak disediakan, set None
    labels_prob = labels_pred if labels_prob is None else labels_prob
    
    # Hitung metrik
    results = calculate_metrics(labels, labels_pred, labels_prob)
    
    # Visualisasi
    # 1. Confusion Matrix
    plot_confusion_matrix(results['confusion_matrix'], 
                          os.path.join(output_dir, "confusion_matrix.png"))
    
    # 2. PCA
    perform_pca(features, labels, n_components=2, 
               output_path=os.path.join(output_dir, "pca_visualization.png"))
    
    # Simpan hasil
    save_evaluation_results(results, os.path.join(output_dir, "evaluation_results.md"))
    
    print("\nEvaluasi sederhana selesai!")
    print(f"Semua hasil disimpan di direktori: {output_dir}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluasi sederhana untuk model AI detector")
    parser.add_argument("--data", type=str, default="dataset/hybrid_features.pkl", 
                        help="Path ke data fitur")
    parser.add_argument("--output", type=str, default="evaluation_results", 
                        help="Direktori output untuk hasil evaluasi")
    
    args = parser.parse_args()
    main(args.data, output_dir=args.output) 