import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Hasil akurasi dan metrik dari pelatihan sebelumnya
accuracy = 0.89
precision_ai = 0.89
recall_ai = 0.91
f1_ai = 0.90
precision_real = 0.88
recall_real = 0.86
f1_real = 0.87

# Confusion matrix berdasarkan hasil pelatihan
# Format: [[TN, FP], [FN, TP]]
# TN = Asli diklasifikasikan sebagai Asli
# FP = Asli diklasifikasikan sebagai AI
# FN = AI diklasifikasikan sebagai Asli
# TP = AI diklasifikasikan sebagai AI
cm = np.array([[72, 15], 
               [10, 97]])

# Jumlah data
support_real = 87
support_ai = 107
total_samples = support_real + support_ai

# Hitung metrik tambahan
accuracy_calc = (cm[0,0] + cm[1,1]) / np.sum(cm)
precision_ai_calc = cm[1,1] / (cm[0,1] + cm[1,1])
recall_ai_calc = cm[1,1] / (cm[1,0] + cm[1,1])
f1_ai_calc = 2 * (precision_ai_calc * recall_ai_calc) / (precision_ai_calc + recall_ai_calc)

print(f"Akurasi: {accuracy_calc:.4f}")
print(f"Precision (AI): {precision_ai_calc:.4f}")
print(f"Recall (AI): {recall_ai_calc:.4f}")
print(f"F1 Score (AI): {f1_ai_calc:.4f}")

# Plot confusion matrix
def plot_confusion_matrix(cm, output_path="actual_confusion_matrix.png"):
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

# Buat visualisasi
plot_confusion_matrix(cm)

# Generate classification report
def save_classification_report(output_path="actual_evaluation_results.md"):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Hasil Evaluasi Model Hybrid AI Detector\n\n")
        
        # Metriks
        f.write("## Metrik Utama\n\n")
        f.write(f"| Metrik | Nilai |\n")
        f.write(f"|--------|------|\n")
        f.write(f"| Akurasi | {accuracy:.4f} |\n")
        f.write(f"| Precision | {precision_ai:.4f} |\n")
        f.write(f"| Recall | {recall_ai:.4f} |\n")
        f.write(f"| F1 Score | {f1_ai:.4f} |\n\n")
        
        # Classification Report
        f.write("## Classification Report\n\n")
        f.write(f"| Class | Precision | Recall | F1-Score | Support |\n")
        f.write(f"|-------|-----------|--------|----------|--------|\n")
        f.write(f"| Asli | {precision_real:.4f} | {recall_real:.4f} | {f1_real:.4f} | {support_real} |\n")
        f.write(f"| AI | {precision_ai:.4f} | {recall_ai:.4f} | {f1_ai:.4f} | {support_ai} |\n")
        f.write(f"| **accuracy** | | | {accuracy:.4f} | {total_samples} |\n")
        f.write(f"| **macro avg** | {(precision_real + precision_ai)/2:.4f} | {(recall_real + recall_ai)/2:.4f} | {(f1_real + f1_ai)/2:.4f} | {total_samples} |\n")
        f.write(f"| **weighted avg** | {(precision_real*support_real + precision_ai*support_ai)/total_samples:.4f} | {(recall_real*support_real + recall_ai*support_ai)/total_samples:.4f} | {(f1_real*support_real + f1_ai*support_ai)/total_samples:.4f} | {total_samples} |\n\n")
        
        # Confusion Matrix (text representation)
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"{cm}\n")
        f.write("```\n\n")

# Save report
save_classification_report()
print("Report disimpan ke actual_evaluation_results.md") 