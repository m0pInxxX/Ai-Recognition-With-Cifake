import os
import numpy as np
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0
from skimage.transform import resize
from utils_feature import extract_all_features

# Cek ketersediaan CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan device: {device}")

class HybridModel(nn.Module):
    def __init__(self, input_shape_features):
        super(HybridModel, self).__init__()
        # CNN backbone
        self.cnn = efficientnet_b0(pretrained=True)
        self.cnn.classifier = nn.Identity()  # Hapus classifier terakhir
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_shape_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 128, 256),  # 1280 adalah output size dari EfficientNetB0
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, cnn_input, features):
        # Process CNN input
        cnn_features = self.cnn(cnn_input)
        
        # Process handcrafted features
        processed_features = self.feature_processor(features)
        
        # Combine features
        combined = torch.cat([cnn_features, processed_features], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        return output

def process_images_and_extract_features(images, extract_features=True):
    """Proses gambar untuk CNN dan ekstrak fitur handcrafted"""
    processed_images = []
    extracted_features = []
    
    for i, img in enumerate(images):
        try:
            # Cek ukuran gambar, resize jika terlalu besar untuk mencegah masalah memori
            max_size = 1024  # Ukuran maksimum untuk sisi terpanjang
            if max(img.shape[0], img.shape[1]) > max_size:
                scale = max_size / max(img.shape[0], img.shape[1])
                new_height = int(img.shape[0] * scale)
                new_width = int(img.shape[1] * scale)
                img = resize(img, (new_height, new_width), anti_aliasing=True, preserve_range=True).astype(img.dtype)
                print(f"Resize gambar {i+1}/{len(images)} dari {img.shape} ke ukuran yang lebih kecil")
            
            # Pra-proses gambar untuk CNN
            # Resize ke ukuran 256x256
            img_resized = resize(img, (256, 256), anti_aliasing=True)
            
            # Konversi ke RGB jika grayscale
            if len(img_resized.shape) == 2:
                img_resized = np.stack([img_resized, img_resized, img_resized], axis=-1)
            elif img_resized.shape[2] == 1:
                img_resized = np.concatenate([img_resized, img_resized, img_resized], axis=-1)
            elif img_resized.shape[2] == 4:
                img_resized = img_resized[:, :, :3]  # Buang channel alpha
            
            # Normalisasi untuk EfficientNet
            img_processed = img_resized * 255.0
            img_processed = torch.from_numpy(img_processed).permute(2, 0, 1).float()  # Convert to CxHxW
            processed_images.append(img_processed)
            
            # Ekstrak fitur handcrafted
            if extract_features:
                try:
                    features = extract_all_features(img, include_cnn=False)
                    extracted_features.append(features)
                except Exception as e:
                    print(f"Error saat mengekstrak fitur dari gambar {i+1}/{len(images)}: {e}")
                    # Tambahkan vektor nol sebagai placeholder
                    if i > 0 and extracted_features:
                        feature_size = len(extracted_features[0])
                        extracted_features.append(np.zeros(feature_size))
                    else:
                        # Jika ini gambar pertama, kita belum tahu ukuran fitur
                        print("Tidak dapat mengekstrak fitur dari gambar pertama. Keluar...")
                        raise e
        except Exception as e:
            print(f"Error memproses gambar {i+1}/{len(images)}: {e}")
            # Skip gambar ini jika terjadi error
            continue
    
    if not processed_images:
        raise ValueError("Tidak ada gambar yang berhasil diproses.")
    
    processed_images = torch.stack(processed_images)
    
    if extract_features:
        extracted_features = np.array(extracted_features)
        return processed_images, extracted_features
    else:
        return processed_images

def load_dataset(dataset_path):
    """Muat dataset dari file pickle atau direktori gambar"""
    try:
        # Cek apakah ada file fitur hybrid yang sudah diekstrak
        hybrid_features_path = os.path.join(dataset_path, 'hybrid_features.pkl')
        if os.path.exists(hybrid_features_path):
            print(f"Menggunakan file fitur yang sudah diekstrak: {hybrid_features_path}")
            with open(hybrid_features_path, 'rb') as f:
                data = pickle.load(f)
                features = data.get('features')
                labels = data.get('labels')
                file_paths = data.get('file_paths', [])
                
                if features is None or labels is None:
                    print("File fitur tidak berisi data yang diperlukan.")
                    return None, None, None
                
                print(f"Berhasil memuat {len(labels)} sampel dari file fitur.")
                print(f"Jumlah sampel kelas positif (AI): {np.sum(labels)}")
                print(f"Jumlah sampel kelas negatif (Real): {len(labels) - np.sum(labels)}")
                
                return features, labels, file_paths
                
        elif dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
                features = data.get('features')
                labels = data.get('labels')
                file_paths = data.get('file_paths', [])
                
                if features is None or labels is None:
                    print("File dataset tidak berisi data yang diperlukan.")
                    return None, None, None
                
                print("Dataset yang diberikan hanya berisi fitur yang sudah diekstrak.")
                print("Untuk model hybrid, diperlukan gambar asli untuk diproses oleh CNN.")
                return None, None, None
        else:
            # Jika path adalah direktori, ekstrak fitur dari gambar
            from extract_features import extract_features_from_directory
            
            # Buat path untuk menyimpan fitur
            features_file = os.path.join(dataset_path, 'hybrid_features.pkl')
            
            if not os.path.exists(features_file):
                print(f"File fitur tidak ditemukan di {features_file}.")
                print("Mengekstrak fitur dari gambar...")
                try:
                    _, labels, file_paths = extract_features_from_directory(dataset_path, features_file)
                except Exception as e:
                    print(f"Error saat mengekstrak fitur: {e}")
                    return None, None, None
            
            try:
                with open(features_file, 'rb') as f:
                    data = pickle.load(f)
                    features = data.get('features')
                    labels = data.get('labels')
                    file_paths = data.get('file_paths', [])
                
                if features is None or labels is None:
                    print("File fitur tidak berisi data yang diperlukan.")
                    return None, None, None
                
                print(f"Berhasil memuat {len(labels)} sampel dari file fitur.")
                print(f"Jumlah sampel kelas positif (AI): {np.sum(labels)}")
                print(f"Jumlah sampel kelas negatif (Real): {len(labels) - np.sum(labels)}")
                
                return features, labels, file_paths
            except Exception as e:
                print(f"Error saat memuat file fitur: {e}")
                return None, None, None
    except Exception as e:
        print(f"Error memuat dataset: {e}")
        return None, None, None

def prepare_data(features, labels, file_paths):
    """Siapkan data untuk model hybrid: ekstrak fitur dan split data"""
    try:
        # Split dataset
        train_indices, test_indices = train_test_split(
            np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
        )
        
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        
        # Muat gambar
        from skimage.io import imread
        
        train_images = []
        test_images = []
        
        # Fungsi untuk memuat gambar dengan penanganan error
        def load_image(path):
            try:
                return imread(path)
            except Exception as e:
                print(f"Error memuat gambar {path}: {e}")
                return None
        
        # Proses gambar latih dalam batch untuk menghemat memori
        print("Memuat gambar latih...")
        batch_size = 20  # Proses 20 gambar sekaligus
        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            print(f"Memproses batch {i//batch_size + 1}/{(len(train_indices)-1)//batch_size + 1}...")
            
            batch_images = []
            for idx in batch_indices:
                img = load_image(file_paths[idx])
                if img is not None:
                    batch_images.append(img)
            
            if batch_images:
                train_images.extend(batch_images)
        
        print("Memuat gambar uji...")
        for i in range(0, len(test_indices), batch_size):
            batch_indices = test_indices[i:i+batch_size]
            print(f"Memproses batch {i//batch_size + 1}/{(len(test_indices)-1)//batch_size + 1}...")
            
            batch_images = []
            for idx in batch_indices:
                img = load_image(file_paths[idx])
                if img is not None:
                    batch_images.append(img)
            
            if batch_images:
                test_images.extend(batch_images)
        
        if not train_images or not test_images:
            print("Tidak ada gambar yang berhasil dimuat.")
            return None, None, None, None, None, None
        
        print(f"Berhasil memuat {len(train_images)} gambar latih dan {len(test_images)} gambar uji.")
        
        # Proses gambar dan ekstrak fitur
        print("Memproses gambar dan mengekstrak fitur...")
        try:
            x_train_cnn, x_train_features = process_images_and_extract_features(train_images)
            x_test_cnn, x_test_features = process_images_and_extract_features(test_images)
        except Exception as e:
            print(f"Error saat memproses gambar dan mengekstrak fitur: {e}")
            return None, None, None, None, None, None
        
        # Standarisasi fitur
        print("Standarisasi fitur...")
        scaler = StandardScaler()
        x_train_features = scaler.fit_transform(x_train_features)
        x_test_features = scaler.transform(x_test_features)
        
        # Konversi ke tensor PyTorch
        x_train_cnn = x_train_cnn.to(device)
        x_test_cnn = x_test_cnn.to(device)
        x_train_features = torch.from_numpy(x_train_features).float().to(device)
        x_test_features = torch.from_numpy(x_test_features).float().to(device)
        y_train = torch.from_numpy(y_train[:len(x_train_features)]).float().to(device)  # Pastikan jumlah label sama dengan fitur
        y_test = torch.from_numpy(y_test[:len(x_test_features)]).float().to(device)
        
        print(f"Data siap: {x_train_cnn.shape[0]} sampel latih, {x_test_cnn.shape[0]} sampel uji")
        
        return x_train_cnn, x_train_features, y_train, x_test_cnn, x_test_features, y_test
    except Exception as e:
        print(f"Error memproses data: {e}")
        return None, None, None, None, None, None

def create_and_train_model(x_train_cnn, x_train_features, y_train, x_test_cnn, x_test_features, y_test, model_save_path):
    """Buat dan latih model hybrid"""
    try:
        # Buat model hybrid
        model = HybridModel(input_shape_features=x_train_features.shape[1])
        model = model.to(device)
        
        print(f"Model dibuat dengan input shape CNN: {x_train_cnn.shape} dan fitur: {x_train_features.shape[1]}")
        
        # Definisikan loss function dan optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        num_epochs = 50
        batch_size = 16  # Kurangi batch size untuk menghemat memori
        best_val_loss = float('inf')
        best_accuracy = 0.0
        patience = 10
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        print(f"Mulai pelatihan dengan batch size {batch_size}, epochs {num_epochs}")
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            batches = 0
            
            # Training dengan mini-batch
            for i in range(0, len(x_train_cnn), batch_size):
                try:
                    batch_cnn = x_train_cnn[i:i+batch_size]
                    batch_features = x_train_features[i:i+batch_size]
                    batch_labels = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_cnn, batch_features)
                    loss = criterion(outputs, batch_labels.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batches += 1
                except Exception as e:
                    print(f"Error saat pelatihan batch {i//batch_size + 1}: {e}")
                    continue
            
            avg_train_loss = total_loss / batches if batches > 0 else float('inf')
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, len(x_test_cnn), batch_size):
                    try:
                        batch_cnn = x_test_cnn[i:i+batch_size]
                        batch_features = x_test_features[i:i+batch_size]
                        batch_labels = y_test[i:i+batch_size]
                        
                        outputs = model(batch_cnn, batch_features)
                        loss = criterion(outputs, batch_labels.unsqueeze(1))
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        predictions = (outputs > 0.5).float()
                        correct += (predictions == batch_labels.unsqueeze(1)).sum().item()
                        total += batch_labels.size(0)
                    except Exception as e:
                        print(f"Error saat validasi batch {i//batch_size + 1}: {e}")
                        continue
            
            avg_val_loss = val_loss / (len(x_test_cnn) // batch_size + 1)
            accuracy = correct / total if total > 0 else 0
            
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(accuracy)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
            
            # Early stopping dan simpan model terbaik
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'accuracy': accuracy,
                    'input_shape_features': x_train_features.shape[1]
                }, model_save_path)
                print(f"Model terbaik disimpan (Val Loss: {best_val_loss:.4f}, Accuracy: {accuracy:.4f})")
            elif accuracy > best_accuracy and avg_val_loss < best_val_loss * 1.1:
                # Juga simpan jika akurasi lebih baik dan loss masih dalam 10% dari yang terbaik
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                    'accuracy': accuracy,
                    'input_shape_features': x_train_features.shape[1]
                }, model_save_path.replace('.pth', '_best_acc.pth'))
                print(f"Model dengan akurasi terbaik disimpan (Accuracy: {accuracy:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Plot learning curves
        try:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Validation Accuracy')
            
            plt.tight_layout()
            plt.savefig(model_save_path.replace('.pth', '_learning_curves.png'))
            print(f"Learning curves disimpan di {model_save_path.replace('.pth', '_learning_curves.png')}")
        except Exception as e:
            print(f"Error saat membuat plot: {e}")
        
        print("Training selesai!")
        print(f"Model terbaik disimpan di {model_save_path} (Val Loss: {best_val_loss:.4f})")
        print(f"Model dengan akurasi terbaik disimpan di {model_save_path.replace('.pth', '_best_acc.pth')} (Accuracy: {best_accuracy:.4f})")
        
        return model
    except Exception as e:
        print(f"Error saat membuat dan melatih model: {e}")
        return None

def train_hybrid_model(dataset_path, model_save_path):
    """Latih model hybrid menggunakan dataset gambar"""
    # Muat dataset
    features, labels, file_paths = load_dataset(dataset_path)
    if features is None:
        return
    
    # Persiapkan data
    x_train_cnn, x_train_features, y_train, x_test_cnn, x_test_features, y_test = prepare_data(features, labels, file_paths)
    if x_train_cnn is None:
        return
    
    # Buat dan latih model
    create_and_train_model(x_train_cnn, x_train_features, y_train, x_test_cnn, x_test_features, y_test, model_save_path)

def main():
    parser = argparse.ArgumentParser(description="Latih model hybrid untuk deteksi gambar AI")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path ke direktori dataset atau file features.pkl")
    parser.add_argument("--model_output", type=str, default="ai_detector_hybrid.pth",
                        help="Path untuk menyimpan model terlatih")
    
    args = parser.parse_args()
    
    print("Melatih model hybrid untuk deteksi gambar AI...")
    train_hybrid_model(args.dataset, args.model_output)
    print("Pelatihan selesai!")

if __name__ == "__main__":
    main() 