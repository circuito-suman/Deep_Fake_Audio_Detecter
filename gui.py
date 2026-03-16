import os
import yaml
import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import your existing feature extractor without changing it
from src.features.extract import FeatureExtractor

class DeepfakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Audio Detection Dashboard")
        self.root.geometry("1100x700")
        self.root.configure(padx=10, pady=10)

        # 1. Load Config and Models
        self.config = self.load_config()
        self.models = self.load_all_models()
        self.extractor = FeatureExtractor(self.config)

        self.audio_path = None

        # 2. Build UI layout
        self.build_ui()

    def load_config(self, config_path='config.yaml'):
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load config.yaml:\n{e}")
            return None

    def load_all_models(self):
        model_dir = self.config['outputs']['model_save_dir']
        loaded_models = {}
        
        if not os.path.exists(model_dir):
            messagebox.showwarning("Warning", f"Model directory not found: {model_dir}")
            return loaded_models

        # Find all .pkl files
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                name = file.replace('.pkl', '').replace('_', ' ').title()
                path = os.path.join(model_dir, file)
                try:
                    loaded_models[name] = joblib.load(path)
                except Exception as e:
                    print(f"Failed to load {file}: {e}")
        
        return loaded_models

    def build_ui(self):
        # --- Top Frame: Controls ---
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.btn_browse = ttk.Button(control_frame, text="Upload Audio File", command=self.browse_file)
        self.btn_browse.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_file = ttk.Label(control_frame, text="No file selected...", font=("Arial", 10, "italic"))
        self.lbl_file.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_analyze = ttk.Button(control_frame, text="Run All Models", command=self.run_analysis, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.RIGHT)

        # --- Main Content Frame ---
        content_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Panel: Visualizations
        self.viz_frame = ttk.LabelFrame(content_frame, text="Audio Visualization")
        content_frame.add(self.viz_frame, weight=3)
        
        # Right Panel: Results
        self.results_frame = ttk.LabelFrame(content_frame, text="Model Comparison")
        content_frame.add(self.results_frame, weight=2)

        # Matplotlib Figure Setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Treeview for Results Table
        columns = ("Model", "Prediction", "Confidence")
        self.tree = ttk.Treeview(self.results_frame, columns=columns, show="headings", height=15)
        self.tree.heading("Model", text="Model")
        self.tree.heading("Prediction", text="Prediction")
        self.tree.heading("Confidence", text="Fake Confidence")
        
        self.tree.column("Model", width=150)
        self.tree.column("Prediction", width=100, anchor=tk.CENTER)
        self.tree.column("Confidence", width=120, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Styling treeview tags
        self.tree.tag_configure('fake', background='#ffcccc') # Light red
        self.tree.tag_configure('real', background='#ccffcc') # Light green

    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.wav *.mp3 *.flac"), ("All Files", "*.*"))
        )
        if filepath:
            self.audio_path = filepath
            self.lbl_file.config(text=os.path.basename(filepath))
            self.btn_analyze.config(state=tk.NORMAL)
            self.plot_audio_basics()

    def plot_audio_basics(self):
        """Plots the waveform immediately upon loading the file."""
        self.ax1.clear()
        self.ax2.clear()
        
        try:
            y, sr = librosa.load(self.audio_path, sr=self.config['features']['sample_rate'])
            
            # Waveform
            librosa.display.waveshow(y, sr=sr, ax=self.ax1, alpha=0.6)
            self.ax1.set_title("Audio Waveform")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.set_ylabel("Amplitude")

            # MFCCs (What the AI sees)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config['features']['n_mfcc'])
            img = librosa.display.specshow(mfccs, x_axis='time', ax=self.ax2)
            self.ax2.set_title("MFCC Features (AI Input)")
            self.ax2.set_xlabel("Time")
            self.ax2.set_ylabel("MFCC Coefficients")

            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Plot Error", f"Could not visualize audio:\n{e}")

    def run_analysis(self):
        if not self.audio_path or not self.models:
            messagebox.showerror("Error", "Missing audio file or trained models.")
            return

        self.btn_analyze.config(text="Processing...", state=tk.DISABLED)
        self.root.update()

        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            # 1. Extract features using your existing code
            features = self.extractor.extract_features(self.audio_path)
            if features is None:
                raise ValueError("Feature extraction returned None.")
            
            X = features.reshape(1, -1)

            # 2. Run through all models
            for model_name, clf in self.models.items():
                prediction = clf.predict(X)[0]
                
                # Try to get probability
                prob_str = "N/A"
                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(X)[0]
                    prob_fake = prob[1] * 100
                    prob_str = f"{prob_fake:.2f}%"
                
                result_text = "FAKE" if prediction == 1 else "REAL"
                tag = 'fake' if prediction == 1 else 'real'

                self.tree.insert("", tk.END, values=(model_name, result_text, prob_str), tags=(tag,))

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
        
        finally:
            self.btn_analyze.config(text="Run All Models", state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorGUI(root)
    root.mainloop()
