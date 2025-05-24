import os
import numpy as np
import pickle
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from skimage.transform import resize
from utils_feature import extract_all_features

def create_hybrid_model(input_shape_cnn=(224, 224, 3), input_shape_features=59):
    """Buat model hybrid yang menggabungkan CNN dan fitur handcrafted"""
    # Input untuk CNN
    cnn_input = Input(shape=input_shape_cnn, name='cnn_input')
    
    # Gunakan EfficientNetB0 yang sudah dilatih pada ImageNet sebagai backbone CNN
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=cnn_input, pooling='avg')
    
    # Freeze beberapa layer awal
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Input untuk fitur handcrafted
    feature_input = Input(shape=(input_shape_features,), name='feature_input')
    
    # Proses fitur handcrafted
    x_feature = Dense(128, activation='relu')(feature_input)
    x_feature = BatchNormalization()(x_feature)
    x_feature = Dropout(0.3)(x_feature)
    
    # Gabungkan output dari CNN dan fitur handcrafted
    combined = Concatenate()([base_model.output, x_feature])
    
    # Tambahkan lapisan klasifikasi
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    output = Dense(1, activation='sigmoid')(x)
    
    # Buat model
    model = Model(inputs=[cnn_input, feature_input], outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def process_images_and_extract_features(images, extract_features=True):
    """Proses gambar untuk CNN dan ekstrak fitur handcrafted"""
    processed_images = []
    extracted_features = []
    
    for img in images:
        # Pra-proses gambar untuk CNN
        # Resize ke ukuran input EfficientNetB0
        img_resized = resize(img, (224, 224), anti_aliasing=True)
        
        # Konversi ke RGB jika grayscale
        if len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized, img_resized, img_resized], axis=-1)
        elif img_resized.shape[2] == 1:
            img_resized = np.concatenate([img_resized, img_resized, img_resized], axis=-1)
        elif img_resized.shape[2] == 4:
            img_resized = img_resized[:, :, :3]  # Buang channel alpha
        
        # Pra-proses untuk EfficientNetB0
        img_processed = tf.keras.applications.efficientnet.preprocess_input(img_resized * 255.0)
        processed_images.append(img_processed)
        
        # Ekstrak fitur handcrafted
        if extract_features:
            features = extract_all_features(img, include_cnn=False)
            extracted_features.append(features)
    
    processed_images = np.array(processed_images)
    
    if extract_features:
        extracted_features = np.array(extracted_features)
        return processed_images, extracted_features
    else:
        return processed_images

def train_hybrid_model(dataset_path, model_save_path):
    """Latih model hybrid menggunakan dataset gambar"""
    # Muat dataset
    try:
        if dataset_path.endswith('.pkl'):
            # Asumsikan ini adalah file fitur, muat fitur yang sudah diekstrak
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']
                file_paths = data.get('file_paths', [])
                
            # Split dataset
            x_train, x_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Untuk model hybrid, kita perlu memuat gambar dan mengekstrak fitur
            # Ini hanya sebagai contoh, Anda perlu menyesuaikan dengan kasus penggunaan sebenarnya
            print("Dataset yang diberikan hanya berisi fitur yang sudah diekstrak.")
            print("Untuk model hybrid, diperlukan gambar asli untuk diproses oleh CNN.")
            print("Keluar...")
            return
        else:
            # Asumsikan ini adalah direktori dengan gambar
            # Muat dan proses gambar
            from extract_features import extract_features_from_directory
            
            # Ekstrak fitur dari direktori
            features_file = os.path.join(os.path.dirname(dataset_path), 'hybrid_features.pkl')
            _, labels, file_paths = extract_features_from_directory(dataset_path, features_file)
            
            # Load fitur yang sudah diekstrak
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
                features = data['features']
                labels = data['labels']
                file_paths = data['file_paths']
            
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
            
            print("Memuat gambar latih...")
            for idx in train_indices:
                img = imread(file_paths[idx])
                train_images.append(img)
            
            print("Memuat gambar uji...")
            for idx in test_indices:
                img = imread(file_paths[idx])
                test_images.append(img)
            
            # Proses gambar dan ekstrak fitur
            print("Memproses gambar dan mengekstrak fitur...")
            x_train_cnn, x_train_features = process_images_and_extract_features(train_images)
            x_test_cnn, x_test_features = process_images_and_extract_features(test_images)
            
            # Normalisasi fitur handcrafted
            scaler = StandardScaler()
            x_train_features = scaler.fit_transform(x_train_features)
            x_test_features = scaler.transform(x_test_features)
            
            # Simpan scaler
            with open('scaler_hybrid.pkl', 'wb') as f:
                pickle.dump(scaler, f)
    except Exception as e:
        print(f"Error memuat atau memproses dataset: {e}")
        return
    
    # Buat model hybrid
    model = create_hybrid_model(input_shape_features=x_train_features.shape[1])
    
    # Siapkan callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy')
    ]
    
    # Latih model
    history = model.fit(
        [x_train_cnn, x_train_features], y_train,
        validation_data=([x_test_cnn, x_test_features], y_test),
        epochs=50,
        batch_size=16,  # Kurangi batch size karena model lebih besar
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluasi model
    y_pred_prob = model.predict([x_test_cnn, x_test_features])
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\nHasil Evaluasi:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Hybrid Model')
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
    plt.savefig('confusion_matrix_hybrid.png')
    
    # Plot kurva akurasi dan loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy - Hybrid')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss - Hybrid')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history_hybrid.png')
    
    # Simpan model
    model.save(model_save_path)
    print(f"Model hybrid disimpan ke {model_save_path}")

def main():
    parser = argparse.ArgumentParser(description="Latih model hybrid untuk deteksi gambar AI")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path ke direktori dataset atau file features.pkl")
    parser.add_argument("--model_output", type=str, default="ai_detector_hybrid.keras",
                        help="Path untuk menyimpan model terlatih")
    
    args = parser.parse_args()
    
    print("Melatih model hybrid untuk deteksi gambar AI...")
    train_hybrid_model(args.dataset, args.model_output)
    print("Pelatihan selesai!")

if __name__ == "__main__":
    main() 