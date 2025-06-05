import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import threading
import time

# Import fungsi dari cifake_classifier.py
from cifake_classifier import load_model_and_scaler, preprocess_image_for_hybrid, is_hybrid_model
from utils_feature import extract_all_features
from skimage.io import imread

class AIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Detector")
        self.root.geometry("1000x600")
        self.root.configure(bg="#f0f0f0")
        
        # Variabel
        self.image_path = ""
        self.image_paths = []  # Untuk batch processing
        self.model_path = ""
        self.scaler_path = ""
        self.model = None
        self.scaler = None
        self.result_label = None
        self.is_processing_batch = False
        
        # Frame Utama
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook untuk tab
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Deteksi Tunggal
        self.single_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_tab, text="Deteksi Tunggal")
        
        # Tab 2: Batch Processing
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="Batch Processing")
        
        # Setup kedua tab
        self.setup_single_tab()
        self.setup_batch_tab()
        
        # Cari model dan scaler default
        self.find_default_models()
    
    def setup_single_tab(self):
        # Frame Utama untuk tab deteksi tunggal
        single_main_frame = ttk.Frame(self.single_tab, padding="10")
        single_main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame Kiri (Kontrol)
        self.control_frame = ttk.Frame(single_main_frame, padding="10", borderwidth=2, relief="groove")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Frame Kanan (Gambar dan Hasil)
        self.display_frame = ttk.Frame(single_main_frame, padding="10")
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_control_panel()
        self.setup_display_panel()
    
    def setup_batch_tab(self):
        # Frame Utama untuk tab batch processing
        batch_main_frame = ttk.Frame(self.batch_tab, padding="10")
        batch_main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame Kiri (Kontrol)
        batch_control_frame = ttk.Frame(batch_main_frame, padding="10", borderwidth=2, relief="groove")
        batch_control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Frame Kanan (Hasil)
        batch_result_frame = ttk.Frame(batch_main_frame, padding="10")
        batch_result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Judul
        title_label = ttk.Label(batch_control_frame, text="Batch Processing", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Pilih Folder atau Beberapa Gambar
        folder_frame = ttk.Frame(batch_control_frame)
        folder_frame.pack(fill=tk.X, pady=10)
        
        folder_button = ttk.Button(folder_frame, text="Pilih Folder", command=self.browse_folder)
        folder_button.pack(fill=tk.X, pady=5)
        
        files_button = ttk.Button(folder_frame, text="Pilih Beberapa Gambar", command=self.browse_multiple_files)
        files_button.pack(fill=tk.X, pady=5)
        
        self.batch_status_var = tk.StringVar(value="Belum ada gambar dipilih")
        status_label = ttk.Label(folder_frame, textvariable=self.batch_status_var, wraplength=200)
        status_label.pack(fill=tk.X, pady=5)
        
        # Pilih Model (reuse dari tab single)
        model_frame = ttk.Frame(batch_control_frame)
        model_frame.pack(fill=tk.X, pady=10)
        
        model_label = ttk.Label(model_frame, text="Model:")
        model_label.pack(anchor=tk.W)
        
        self.batch_model_var = tk.StringVar()
        self.batch_model_dropdown = ttk.Combobox(model_frame, textvariable=self.batch_model_var)
        self.batch_model_dropdown.pack(fill=tk.X, pady=5)
        
        # Pilih Scaler (reuse dari tab single)
        scaler_frame = ttk.Frame(batch_control_frame)
        scaler_frame.pack(fill=tk.X, pady=10)
        
        scaler_label = ttk.Label(scaler_frame, text="Scaler:")
        scaler_label.pack(anchor=tk.W)
        
        self.batch_scaler_var = tk.StringVar()
        self.batch_scaler_dropdown = ttk.Combobox(scaler_frame, textvariable=self.batch_scaler_var)
        self.batch_scaler_dropdown.pack(fill=tk.X, pady=5)
        
        # Opsi Export
        export_frame = ttk.LabelFrame(batch_control_frame, text="Opsi Export", padding=10)
        export_frame.pack(fill=tk.X, pady=10)
        
        self.export_csv_var = tk.BooleanVar(value=True)
        csv_check = ttk.Checkbutton(export_frame, text="Export ke CSV", variable=self.export_csv_var)
        csv_check.pack(anchor=tk.W, pady=5)
        
        # Tombol Proses Batch
        process_button = ttk.Button(batch_control_frame, text="Proses Batch", 
                                    command=self.process_batch, style="Accent.TButton")
        process_button.pack(fill=tk.X, pady=20)
        
        # Progress Bar
        self.batch_progress_var = tk.DoubleVar(value=0)
        self.batch_progress_bar = ttk.Progressbar(batch_control_frame, orient=tk.HORIZONTAL, 
                                                 length=200, mode='determinate', 
                                                 variable=self.batch_progress_var)
        self.batch_progress_bar.pack(fill=tk.X, pady=10)
        
        self.batch_progress_text = tk.StringVar(value="0%")
        progress_label = ttk.Label(batch_control_frame, textvariable=self.batch_progress_text)
        progress_label.pack(pady=5)
        
        # Tabel hasil
        columns = ("filename", "prediction", "probability")
        self.result_tree = ttk.Treeview(batch_result_frame, columns=columns, show="headings")
        
        # Setup headings
        self.result_tree.heading("filename", text="Nama File")
        self.result_tree.heading("prediction", text="Prediksi")
        self.result_tree.heading("probability", text="Probabilitas")
        
        # Setup columns
        self.result_tree.column("filename", width=200)
        self.result_tree.column("prediction", width=100)
        self.result_tree.column("probability", width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(batch_result_frame, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def setup_control_panel(self):
        # Judul
        title_label = ttk.Label(self.control_frame, text="AI Image Detector", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Pilih Gambar
        img_frame = ttk.Frame(self.control_frame)
        img_frame.pack(fill=tk.X, pady=10)
        
        img_label = ttk.Label(img_frame, text="Pilih Gambar:")
        img_label.pack(anchor=tk.W)
        
        img_button = ttk.Button(img_frame, text="Browse...", command=self.browse_image)
        img_button.pack(fill=tk.X, pady=5)
        
        self.image_path_var = tk.StringVar()
        img_path_label = ttk.Label(img_frame, textvariable=self.image_path_var, wraplength=200)
        img_path_label.pack(fill=tk.X)
        
        # Pilih Model
        model_frame = ttk.Frame(self.control_frame)
        model_frame.pack(fill=tk.X, pady=10)
        
        model_label = ttk.Label(model_frame, text="Model:")
        model_label.pack(anchor=tk.W)
        
        self.model_path_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_path_var)
        self.model_dropdown.pack(fill=tk.X, pady=5)
        
        model_button = ttk.Button(model_frame, text="Browse Model...", command=self.browse_model)
        model_button.pack(fill=tk.X)
        
        # Pilih Scaler
        scaler_frame = ttk.Frame(self.control_frame)
        scaler_frame.pack(fill=tk.X, pady=10)
        
        scaler_label = ttk.Label(scaler_frame, text="Scaler:")
        scaler_label.pack(anchor=tk.W)
        
        self.scaler_path_var = tk.StringVar()
        self.scaler_dropdown = ttk.Combobox(scaler_frame, textvariable=self.scaler_path_var)
        self.scaler_dropdown.pack(fill=tk.X, pady=5)
        
        scaler_button = ttk.Button(scaler_frame, text="Browse Scaler...", command=self.browse_scaler)
        scaler_button.pack(fill=tk.X)
        
        # Tombol Deteksi
        detect_button = ttk.Button(self.control_frame, text="Deteksi Gambar", command=self.detect_image, style="Accent.TButton")
        detect_button.pack(fill=tk.X, pady=20)
        
        # Hasil
        result_frame = ttk.LabelFrame(self.control_frame, text="Hasil Deteksi", padding=10)
        result_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Menggunakan Text widget alih-alih Label untuk hasil yang lebih baik
        self.result_text = tk.Text(result_frame, height=4, width=25, font=("Arial", 12))
        self.result_text.pack(pady=5, fill=tk.X)
        self.result_text.configure(state='disabled')  # Buat read-only
        
        # Progress Bar dengan margin yang lebih baik
        self.progress_bar = ttk.Progressbar(self.control_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=10, padx=5)
    
    def setup_display_panel(self):
        # Frame untuk gambar
        self.image_frame = ttk.Frame(self.display_frame, borderwidth=2, relief="groove")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Label untuk menampilkan gambar
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label informasi
        info_label = ttk.Label(self.display_frame, text="Pilih gambar untuk dideteksi", font=("Arial", 10))
        info_label.pack(pady=5)
    
    def find_default_models(self):
        """Mencari model dan scaler default"""
        models = []
        scalers = []
        
        # Cari file dengan ekstensi .keras atau .h5
        for file in os.listdir('.'):
            if file.endswith('.keras') or file.endswith('.h5'):
                models.append(file)
            elif file.endswith('.pkl') and 'scaler' in file.lower():
                scalers.append(file)
        
        # Update dropdown models
        if models:
            self.model_dropdown['values'] = models
            self.model_dropdown.current(0)
            self.model_path = models[0]
            self.model_path_var.set(models[0])
            
            # Update untuk batch tab juga
            self.batch_model_dropdown['values'] = models
            self.batch_model_dropdown.current(0)
            self.batch_model_var.set(models[0])
        
        # Update dropdown scalers
        if scalers:
            self.scaler_dropdown['values'] = scalers
            self.scaler_dropdown.current(0)
            self.scaler_path = scalers[0]
            self.scaler_path_var.set(scalers[0])
            
            # Update untuk batch tab juga
            self.batch_scaler_dropdown['values'] = scalers
            self.batch_scaler_dropdown.current(0)
            self.batch_scaler_var.set(scalers[0])
    
    def browse_image(self):
        """Buka dialog untuk memilih gambar"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        image_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=filetypes
        )
        
        if image_path:
            self.image_path = image_path
            self.image_path_var.set(os.path.basename(image_path))
            self.load_and_display_image(image_path)
    
    def browse_folder(self):
        """Buka dialog untuk memilih folder"""
        folder_path = filedialog.askdirectory(
            title="Pilih Folder yang Berisi Gambar"
        )
        
        if folder_path:
            # Cari semua file gambar di folder
            self.image_paths = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(folder_path, file))
            
            if self.image_paths:
                self.batch_status_var.set(f"Ditemukan {len(self.image_paths)} gambar di {os.path.basename(folder_path)}")
            else:
                self.batch_status_var.set(f"Tidak ditemukan gambar di {os.path.basename(folder_path)}")
    
    def browse_multiple_files(self):
        """Buka dialog untuk memilih beberapa gambar"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        file_paths = filedialog.askopenfilenames(
            title="Pilih Beberapa Gambar",
            filetypes=filetypes
        )
        
        if file_paths:
            self.image_paths = file_paths
            self.batch_status_var.set(f"Dipilih {len(self.image_paths)} gambar")
    
    def browse_model(self):
        """Buka dialog untuk memilih model"""
        filetypes = [
            ("Keras model", "*.keras *.h5"),
            ("All files", "*.*")
        ]
        
        model_path = filedialog.askopenfilename(
            title="Pilih Model",
            filetypes=filetypes
        )
        
        if model_path:
            self.model_path = model_path
            self.model_path_var.set(os.path.basename(model_path))
            # Reset model dan scaler
            self.model = None
            self.scaler = None
    
    def browse_scaler(self):
        """Buka dialog untuk memilih scaler"""
        filetypes = [
            ("Pickle files", "*.pkl"),
            ("All files", "*.*")
        ]
        
        scaler_path = filedialog.askopenfilename(
            title="Pilih Scaler",
            filetypes=filetypes
        )
        
        if scaler_path:
            self.scaler_path = scaler_path
            self.scaler_path_var.set(os.path.basename(scaler_path))
            # Reset scaler
            self.scaler = None
    
    def load_and_display_image(self, image_path):
        """Muat dan tampilkan gambar"""
        try:
            # Buka gambar dengan PIL
            img = Image.open(image_path)
            
            # Resize agar pas di frame, jaga rasio aspek
            width, height = img.size
            max_size = min(self.image_frame.winfo_width(), self.image_frame.winfo_height())
            
            if max_size == 0:  # Jika frame belum dirender
                max_size = 400
            
            # Hitung rasio ukuran gambar
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            # Resize gambar
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Konversi ke format Tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Tampilkan gambar
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Simpan referensi
            
            # Reset hasil
            self.result_text.configure(state='normal')  # Buka akses untuk edit
            self.result_text.delete(1.0, tk.END)  # Hapus konten sebelumnya
            self.result_text.insert(tk.END, f"Hasil Deteksi:\n\n")
            self.result_text.insert(tk.END, f"AI: 0%\n")
            self.result_text.insert(tk.END, f"Asli: 0%")
            self.result_text.configure(state='disabled')  # Kunci kembali
            
        except Exception as e:
            messagebox.showerror("Error", f"Tidak dapat memuat gambar: {str(e)}")
    
    def detect_image(self):
        """Deteksi gambar menggunakan model"""
        if not self.image_path:
            messagebox.showerror("Error", "Pilih gambar terlebih dahulu!")
            return
        
        if not self.model_path:
            messagebox.showerror("Error", "Pilih model terlebih dahulu!")
            return
        
        try:
            # Tampilkan progress bar
            self.progress_bar.start()
            
            # Muat model dan scaler jika belum dimuat
            if self.model is None or self.scaler is None:
                self.model, self.scaler = load_model_and_scaler(self.model_path, self.scaler_path)
                
                if self.model is None:
                    messagebox.showerror("Error", "Gagal memuat model!")
                    self.progress_bar.stop()
                    return
            
            # Baca gambar
            image = imread(self.image_path)
            
            # Pastikan gambar dalam format yang benar
            # Jika gambar memiliki 4 channel (RGBA), ambil hanya 3 channel (RGB)
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            
            # Cek apakah model hybrid atau bukan
            if is_hybrid_model(self.model):
                # Pra-proses untuk hybrid model
                img_cnn, features = preprocess_image_for_hybrid(image)
                
                # Pra-proses fitur dengan scaler jika ada
                if self.scaler:
                    features = self.scaler.transform(features)
                
                # Prediksi
                prediction_prob = self.model.predict([img_cnn, features])[0][0]
            else:
                # Ekstrak fitur untuk model tradisional
                features = extract_all_features(image)
                
                # Reshape features untuk prediksi
                features = features.reshape(1, -1)
                
                # Pra-proses fitur jika scaler tersedia
                if self.scaler:
                    features = self.scaler.transform(features)
                
                # Prediksi
                prediction_prob = self.model.predict(features)[0][0]
            
            # Tampilkan hasil
            self.result_text.configure(state='normal')  # Buka akses untuk edit
            self.result_text.delete(1.0, tk.END)  # Hapus konten sebelumnya
            self.result_text.insert(tk.END, f"Hasil Deteksi:\n\n")
            self.result_text.insert(tk.END, f"AI: {prediction_prob:.2%}\n")
            self.result_text.insert(tk.END, f"Asli: {(1 - prediction_prob):.2%}")
            self.result_text.configure(state='disabled')  # Kunci kembali
            
            # Hentikan progress bar
            self.progress_bar.stop()
            
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat deteksi: {str(e)}")
            self.progress_bar.stop()
    
    def process_single_image(self, image_path, model, scaler):
        """Proses satu gambar untuk batch processing"""
        try:
            # Baca gambar
            image = imread(image_path)
            
            # Pastikan gambar dalam format yang benar
            # Jika gambar memiliki 4 channel (RGBA), ambil hanya 3 channel (RGB)
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            
            # Cek apakah model hybrid atau bukan
            if is_hybrid_model(model):
                # Pra-proses untuk hybrid model
                img_cnn, features = preprocess_image_for_hybrid(image)
                
                # Pra-proses fitur dengan scaler jika ada
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
            
            # Tentukan kelas prediksi
            prediction_class = 'AI' if prediction_prob > 0.5 else 'Asli'
            
            return os.path.basename(image_path), prediction_class, prediction_prob
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return os.path.basename(image_path), "Error", 0.0
    
    def process_batch(self):
        """Proses batch gambar"""
        if not self.image_paths:
            messagebox.showerror("Error", "Pilih gambar terlebih dahulu!")
            return
        
        model_path = self.batch_model_var.get()
        scaler_path = self.batch_scaler_var.get()
        
        if not model_path:
            messagebox.showerror("Error", "Pilih model terlebih dahulu!")
            return
        
        # Jika sudah ada proses batch yang berjalan, jangan mulai lagi
        if self.is_processing_batch:
            messagebox.showinfo("Info", "Batch processing sedang berjalan")
            return
        
        # Menandai bahwa proses batch sedang berjalan
        self.is_processing_batch = True
        
        # Mulai thread baru untuk proses batch
        thread = threading.Thread(target=self._run_batch_processing)
        thread.daemon = True  # Thread akan dimatikan ketika program utama selesai
        thread.start()
    
    def _run_batch_processing(self):
        """Jalankan batch processing di thread terpisah"""
        try:
            # Kosongkan hasil sebelumnya
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            
            # Reset progress
            self.batch_progress_var.set(0)
            self.batch_progress_text.set("0%")
            
            # Muat model dan scaler
            model_path = os.path.join(os.getcwd(), self.batch_model_var.get())
            scaler_path = os.path.join(os.getcwd(), self.batch_scaler_var.get()) if self.batch_scaler_var.get() else None
            
            model, scaler = load_model_and_scaler(model_path, scaler_path)
            
            if model is None:
                messagebox.showerror("Error", "Gagal memuat model!")
                self.is_processing_batch = False
                return
            
            # Siapkan untuk hasil
            results = []
            total_images = len(self.image_paths)
            
            # Proses setiap gambar
            for i, image_path in enumerate(self.image_paths):
                # Update progress
                progress = (i / total_images) * 100
                self.batch_progress_var.set(progress)
                self.batch_progress_text.set(f"{int(progress)}%")
                
                # Proses gambar
                filename, prediction, probability = self.process_single_image(image_path, model, scaler)
                
                # Tambahkan ke hasil
                results.append((filename, prediction, probability))
                
                # Tambahkan ke treeview
                self.result_tree.insert('', 'end', values=(
                    filename, 
                    prediction, 
                    f"{probability:.3%}" if isinstance(probability, float) else probability
                ))
                
                # Update GUI
                self.root.update()
            
            # Selesai
            self.batch_progress_var.set(100)
            self.batch_progress_text.set("100%")
            
            # Export hasil jika diminta
            if self.export_csv_var.get():
                self.export_results_to_csv(results)
            
            messagebox.showinfo("Sukses", f"Berhasil memproses {total_images} gambar")
        
        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat batch processing: {str(e)}")
        
        finally:
            # Tandai bahwa proses batch telah selesai
            self.is_processing_batch = False
    
    def export_results_to_csv(self, results):
        """Export hasil ke file CSV"""
        try:
            # Buat DataFrame
            df = pd.DataFrame(results, columns=['filename', 'prediction', 'probability'])
            
            # Buka dialog untuk memilih lokasi penyimpanan
            file_path = filedialog.asksaveasfilename(
                title="Simpan CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Simpan ke CSV
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Sukses", f"Hasil disimpan ke {file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan hasil: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIDetectorGUI(root)
    
    # Style
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="black", background="#4CAF50", font=("Arial", 10, "bold"))
    
    root.mainloop() 