import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import pickle
from skimage.io import imread
from utils_feature import extract_all_features
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def extract_single_feature(args):
    img_path, label, include_cnn = args
    try:
        image = imread(img_path)
        feature_vector = extract_all_features(image, include_cnn=include_cnn)
        return feature_vector, label, img_path, None
    except Exception as e:
        return None, label, img_path, str(e)

def extract_features_from_directory(directory, output_file, include_cnn=True, num_workers=None):
    """Ekstrak fitur dari semua gambar dalam direktori dengan multiprocessing"""
    features = []
    labels = []
    file_paths = []
    errors = []

    # Cek jika ada subdirektori (kelas) atau tidak
    sub_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    tasks = []
    if sub_dirs:
        for class_dir in sub_dirs:
            class_path = os.path.join(directory, class_dir)
            if 'ai' in class_dir.lower() or 'fake' in class_dir.lower() or 'generated' in class_dir.lower():
                label = 1
            else:
                label = 0
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                tasks.append((img_path, label, include_cnn))
    else:
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)
            if 'ai' in img_file.lower() or 'fake' in img_file.lower() or 'generated' in img_file.lower():
                label = 1
            else:
                label = 0
            tasks.append((img_path, label, include_cnn))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    print(f"Menggunakan {num_workers} proses untuk ekstraksi fitur...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(extract_single_feature, task): i for i, task in enumerate(tasks)}
        results = [None] * len(tasks)
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Ekstraksi fitur (multiprocessing)"):
            i = futures[future]
            try:
                feature_vector, label, img_path, error = future.result()
                if feature_vector is not None:
                    results[i] = (feature_vector, label, img_path)
                else:
                    errors.append((img_path, error))
            except Exception as e:
                errors.append((tasks[i][0], str(e)))

    # Filter hasil sukses
    for res in results:
        if res is not None:
            feature_vector, label, img_path = res
            features.append(feature_vector)
            labels.append(label)
            file_paths.append(img_path)

    features = np.array(features)
    labels = np.array(labels)

    with open(output_file, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels, 'file_paths': file_paths}, f)

    print(f"Fitur disimpan ke {output_file}")
    print(f"Jumlah gambar: {len(labels)}")
    print(f"Jumlah gambar AI: {np.sum(labels)}")
    print(f"Jumlah gambar asli: {len(labels) - np.sum(labels)}")
    if errors:
        print(f"{len(errors)} gambar gagal diekstrak. Cek log error jika perlu.")
    return features, labels, file_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstrak fitur dari dataset gambar")
    parser.add_argument("--input_dir", type=str, required=True, help="Direktori input dengan gambar")
    parser.add_argument("--output_file", type=str, default="features.pkl", help="File output untuk fitur")
    parser.add_argument("--include_cnn", action="store_true", help="Sertakan fitur CNN")
    parser.add_argument("--num_workers", type=int, default=None, help="Jumlah proses paralel (default: semua core)")
    args = parser.parse_args()

    extract_features_from_directory(args.input_dir, args.output_file, args.include_cnn, args.num_workers) 