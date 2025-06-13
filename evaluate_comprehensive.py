import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from train_hybrid import HybridModel
import pandas as pd
from skimage.io import imread

# Cek ketersediaan CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan device: {device}")

def load_model_and_data(model_path, data_path):
    """Muat model dan data uji"""
    # Muat data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data.get('features')
    labels = data.get('labels')
    file_paths = data.get('file_paths', [])
    
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Total sampel: {len(labels)}")
    
    # Muat model
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        num_features = checkpoint['input_shape_features']
        model = HybridModel(input_shape_features=num_features)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model dimuat dengan input_shape_features={num_features}")
    else:
        input_shape_features = features.shape[1]
        model = HybridModel(input_shape_features=input_shape_features)
        model.load_state_dict(checkpoint)
        print(f"Model dimuat dari state dict, menggunakan input_shape_features={input_shape_features}")
    
    model = model.to(device)
    model.eval()
    
    return model, features, labels, file_paths

def get_predictions(model, features):
    """Dapatkan prediksi dari model"""
    # Convert data ke tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Batch processing untuk menghindari OOM
    batch_size = 32
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        # Placeholder untuk input CNN
        dummy_images = torch.zeros((features_tensor.shape[0], 3, 256, 256), device=device)
        
        # Debug informasi
        print(f"Shape features tensor: {features_tensor.shape}")
        print(f"Shape dummy images: {dummy_images.shape}")
        
        # Process data in batches
        for i in range(0, len(features_tensor), batch_size):
            end_idx = min(i + batch_size, len(features_tensor))
            batch_features = features_tensor[i:end_idx]
            batch_images = dummy_images[i:end_idx]
            
            # Debug
            if i == 0:
                print(f"Batch features shape: {batch_features.shape}")
                print(f"Batch images shape: {batch_images.shape}")
            
            try:
                outputs = model(batch_images, batch_features)
                preds = (outputs > 0.5).int().squeeze().cpu().numpy()
                probs = outputs.squeeze().cpu().numpy()
                
                # Handle single output case
                if not isinstance(preds, np.ndarray):
                    preds = np.array([preds])
                    probs = np.array([probs])
                    
                all_preds.extend(preds)
                all_probs.extend(probs)
            except Exception as e:
                print(f"Error pada batch {i}-{end_idx}: {e}")
                # Check model architecture
                print("\nModel architecture:")
                for name, param in model.named_parameters():
                    if 'feature_processor' in name and 'weight' in name:
                        print(f"{name}: {param.shape}")
                raise
    
    return np.array(all_preds), np.array(all_probs)

def evaluate_model(y_true, y_pred, y_prob):
    """Evaluasi model dan hitung metrik"""
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
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    return {
        'report': report,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'pr_curve': {'precision': precision_curve, 'recall': recall_curve, 'auc': pr_auc}
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

def plot_roc_curve(fpr, tpr, roc_auc, output_path="roc_curve.png"):
    """Plot dan simpan ROC curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve disimpan ke {output_path}")

def plot_precision_recall_curve(precision, recall, pr_auc, output_path="precision_recall_curve.png"):
    """Plot dan simpan Precision-Recall curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve disimpan ke {output_path}")

def plot_feature_importance(feature_weights, output_path="feature_importance.png"):
    """Plot dan simpan feature importance"""
    if feature_weights is None:
        print("Feature weights tidak tersedia")
        return
    
    # Hitung importance (perkiraan berdasarkan bobot absolut)
    if len(feature_weights.shape) > 1:
        importance = np.abs(feature_weights).mean(axis=0)
    else:
        importance = np.abs(feature_weights)
    
    # Pilih top N fitur berdasarkan importance
    top_n = min(20, len(importance))
    indices = np.argsort(importance)[::-1][:top_n]
    top_importance = importance[indices]
    feature_names = [f"Feature {i+1}" for i in indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_importance)), top_importance, align='center')
    plt.yticks(range(len(top_importance)), feature_names)
    plt.xlabel('Importance (estimasi dari bobot)')
    plt.ylabel('Fitur')
    plt.title('Estimasi Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance disimpan ke {output_path}")

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
        f.write(f"| ROC AUC | {results['roc']['auc']:.4f} |\n")
        f.write(f"| PR AUC | {results['pr_curve']['auc']:.4f} |\n\n")
        
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
        f.write("2. `roc_curve.png` - ROC Curve\n")
        f.write("3. `precision_recall_curve.png` - Precision-Recall Curve\n")
        f.write("4. `pca_visualization.png` - PCA Visualization\n")
        f.write("5. `feature_importance.png` - Feature Importance\n")
    
    print(f"Hasil evaluasi disimpan ke {output_path}")

def main(model_path, data_path, output_dir="."):
    """Fungsi utama untuk evaluasi komprehensif"""
    # Pastikan output directory ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Muat model dan data
    model, features, labels, file_paths = load_model_and_data(model_path, data_path)
    
    # Dapatkan prediksi
    y_pred, y_prob = get_predictions(model, features)
    
    # Evaluasi
    results = evaluate_model(labels, y_pred, y_prob)
    
    # Visualisasi
    # 1. Confusion Matrix
    plot_confusion_matrix(results['confusion_matrix'], 
                          os.path.join(output_dir, "confusion_matrix.png"))
    
    # 2. ROC Curve
    plot_roc_curve(results['roc']['fpr'], results['roc']['tpr'], results['roc']['auc'],
                  os.path.join(output_dir, "roc_curve.png"))
    
    # 3. Precision-Recall Curve
    plot_precision_recall_curve(results['pr_curve']['precision'], results['pr_curve']['recall'], 
                               results['pr_curve']['auc'],
                               os.path.join(output_dir, "precision_recall_curve.png"))
    
    # 4. PCA
    perform_pca(features, labels, n_components=2, 
               output_path=os.path.join(output_dir, "pca_visualization.png"))
    
    # 5. Feature Importance
    # Dapatkan bobot dari model
    feature_weights = None
    for name, param in model.state_dict().items():
        if 'feature_processor' in name and 'weight' in name:
            feature_weights = param.cpu().numpy()
            break
    
    plot_feature_importance(feature_weights, output_path=os.path.join(output_dir, "feature_importance.png"))
    
    # Simpan hasil
    save_evaluation_results(results, os.path.join(output_dir, "evaluation_results.md"))
    
    print("\nEvaluasi komprehensif selesai!")
    print(f"Semua hasil disimpan di direktori: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluasi komprehensif model AI detector")
    parser.add_argument("--model", type=str, default="ai_detector_hybrid_best_acc.pth", 
                        help="Path ke model terlatih")
    parser.add_argument("--data", type=str, default="dataset/hybrid_features.pkl", 
                        help="Path ke data uji")
    parser.add_argument("--output", type=str, default="evaluation_results", 
                        help="Direktori output untuk hasil evaluasi")
    
    args = parser.parse_args()
    main(args.model, args.data, args.output) 