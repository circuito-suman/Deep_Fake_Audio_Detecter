import os
import sys
import warnings

# ==========================================
# HARDWARE OPTIMIZATIONS & CRASH PREVENTION
# ==========================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["XGBOOST_VERBOSITY"] = "0"
warnings.filterwarnings("ignore")

import yaml
import joblib
import numpy as np
import librosa
import librosa.display
import xgboost as xgb
xgb.set_config(verbosity=0)

import matplotlib.pyplot as plt
# Force Matplotlib graphs to use a dark theme to match the GUI
plt.style.use('dark_background')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QProgressBar, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QTabWidget, QSplitter, QTabBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

from src.features.extract import FeatureExtractor

# ==========================================
# PARALLEL WORKER THREAD
# ==========================================
class AudioWorker(QThread):
    progress = pyqtSignal(int, str)
    audio_visuals_ready = pyqtSignal(np.ndarray, int, np.ndarray)
    result_ready = pyqtSignal(str, str, float, float)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, audio_path, config):
        super().__init__()
        self.audio_path = audio_path
        self.config = config

    def run(self):
        try:
            self.progress.emit(10, "Allocating models to hardware...")
            models = {}
            model_dir = self.config.get('outputs', {}).get('model_save_dir', 'src/models/saved')
            
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pkl'):
                        name = file.replace('.pkl', '').replace('_', ' ').title()
                        path = os.path.join(model_dir, file)
                        try:
                            clf = joblib.load(path)
                            if hasattr(clf, "set_params"):
                                try:
                                    clf.set_params(device="cpu", n_jobs=1)
                                except Exception:
                                    pass
                            models[name] = clf
                        except Exception:
                            pass

            self.progress.emit(30, "Loading audio file...")
            y, sr = librosa.load(self.audio_path, sr=self.config['features']['sample_rate'])
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config['features']['n_mfcc'])
            self.audio_visuals_ready.emit(y, sr, mfccs)

            self.progress.emit(50, "Extracting Deep Learning Features...")
            extractor = FeatureExtractor(self.config)
            features = extractor.extract_features(self.audio_path)
            
            if features is None:
                raise ValueError("Feature extraction failed.")
            
            X = features.reshape(1, -1).astype(np.float32)
            
            total_models = len(models)
            step = 50 / total_models if total_models > 0 else 0
            current_progress = 50

            for model_name, clf in models.items():
                self.progress.emit(int(current_progress), f"Analyzing with {model_name}...")
                
                prediction = clf.predict(X)[0]
                prob_fake, prob_real = 0.0, 0.0
                
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X)[0]
                    prob_fake = probs[1]
                    prob_real = probs[0]
                
                result_label = "FAKE" if prediction == 1 else "REAL"
                confidence = prob_fake if prediction == 1 else prob_real

                self.result_ready.emit(model_name, result_label, confidence * 100, prob_fake * 100)
                current_progress += step

            self.progress.emit(100, "Analysis Complete!")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


# ==========================================
# INDIVIDUAL AUDIO TAB WIDGET
# ==========================================
class AudioAnalysisTab(QWidget):
    def __init__(self, audio_path, config):
        super().__init__()
        self.audio_path = audio_path
        self.config = config
        
        self.model_names_for_chart = []
        self.fake_probs_for_chart = []
        
        self.init_ui()
        self.start_analysis()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        status_layout = QHBoxLayout()
        self.lbl_filename = QLabel(f"File: {os.path.basename(self.audio_path)}")
        self.lbl_filename.setFont(QFont("Arial", 12, QFont.Bold))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.lbl_status = QLabel("Initializing...")
        
        status_layout.addWidget(self.lbl_filename)
        status_layout.addStretch()
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar, 2)
        main_layout.addLayout(status_layout)

        splitter = QSplitter(Qt.Horizontal)
        
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        # Apply dark background to the Matplotlib Figure directly
        self.audio_figure = Figure(figsize=(5, 5), facecolor='#2b2b2b')
        self.audio_canvas = FigureCanvas(self.audio_figure)
        viz_layout.addWidget(self.audio_canvas)
        splitter.addWidget(viz_widget)

        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Model", "Prediction", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.table, 1)

        self.chart_figure = Figure(figsize=(4, 3), facecolor='#2b2b2b')
        self.chart_canvas = FigureCanvas(self.chart_figure)
        self.chart_ax = self.chart_figure.add_subplot(111)
        self.chart_ax.set_ylim(0, 100)
        self.chart_ax.set_ylabel("Fake Probability %", color='white')
        self.chart_ax.tick_params(colors='white')
        results_layout.addWidget(self.chart_canvas, 1)
        
        splitter.addWidget(results_widget)
        main_layout.addWidget(splitter)

    def start_analysis(self):
        self.worker = AudioWorker(self.audio_path, self.config)
        self.worker.progress.connect(self.update_progress)
        self.worker.audio_visuals_ready.connect(self.plot_audio)
        self.worker.result_ready.connect(self.add_result)
        self.worker.finished.connect(self.on_finish)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(msg)

    def plot_audio(self, y, sr, mfccs):
        self.audio_figure.clear()
        ax1 = self.audio_figure.add_subplot(211)
        ax2 = self.audio_figure.add_subplot(212)
        
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#00d2ff')
        ax1.set_title("Waveform (Time Domain)", color='white')
        ax1.tick_params(colors='white')
        
        librosa.display.specshow(mfccs, x_axis='time', ax=ax2, cmap='magma')
        ax2.set_title("MFCC Features (AI Input)", color='white')
        ax2.tick_params(colors='white')
        
        self.audio_figure.tight_layout()
        self.audio_canvas.draw()

    def add_result(self, model_name, result_label, confidence, prob_fake):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        item_model = QTableWidgetItem(model_name)
        item_pred = QTableWidgetItem(result_label)
        item_conf = QTableWidgetItem(f"{confidence:.1f}%")
        
        for item in [item_model, item_pred, item_conf]:
            item.setTextAlignment(Qt.AlignCenter)
            
        # Use darker, more readable colors for the table cells in dark mode
        bg_color = QColor(139, 0, 0) if result_label == "FAKE" else QColor(0, 100, 0)
        for item in [item_model, item_pred, item_conf]:
            item.setBackground(bg_color)
            item.setForeground(QColor("white")) # Force text to be white inside the colored cell
            self.table.setItem(row, item.column(), item)
            
        self.model_names_for_chart.append(model_name)
        self.fake_probs_for_chart.append(prob_fake)
        self.chart_ax.clear()
        
        colors = ['#ff4c4c' if p > 50 else '#4caf50' for p in self.fake_probs_for_chart]
        bars = self.chart_ax.bar(self.model_names_for_chart, self.fake_probs_for_chart, color=colors)
        
        self.chart_ax.set_ylim(0, 100)
        self.chart_ax.set_ylabel("Fake Probability (%)", color='white')
        self.chart_ax.tick_params(colors='white')
        self.chart_ax.axhline(50, color='white', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            # Force chart text to be white
            self.chart_ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=8, color='white')

        self.chart_figure.tight_layout()
        self.chart_canvas.draw()

    def on_finish(self):
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4caf50; }")

    def on_error(self, msg):
        self.lbl_status.setText(f"Error: {msg}")
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff4c4c; }")


# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
class DeepfakeStudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deepfake Audio Studio - Multi-Core Edition")
        self.setGeometry(100, 100, 1300, 800)
        self.config = self.load_config()
        self.init_ui()

    def load_config(self, config_path='config.yaml'):
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception:
            return {}

    def init_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabs)

        self.home_tab = QWidget()
        home_layout = QVBoxLayout(self.home_tab)
        
        welcome_lbl = QLabel("Deepfake Audio Detection Studio")
        welcome_lbl.setFont(QFont("Arial", 24, QFont.Bold))
        welcome_lbl.setAlignment(Qt.AlignCenter)

        self.btn_select = QPushButton("Select Audio Files (Ctrl+Click for multiple)")
        self.btn_select.setFixedSize(400, 60)
        self.btn_select.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_select.clicked.connect(self.add_audio_files)

        home_layout.addStretch()
        home_layout.addWidget(welcome_lbl)
        home_layout.addSpacing(20)
        home_layout.addWidget(self.btn_select, alignment=Qt.AlignCenter)
        home_layout.addStretch()

        self.tabs.addTab(self.home_tab, "🏠 Home")
        self.tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)

    def add_audio_files(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)", options=options)
        
        for path in file_paths:
            new_tab = AudioAnalysisTab(path, self.config)
            filename = os.path.basename(path)
            tab_index = self.tabs.addTab(new_tab, f"🎵 {filename}")
            self.tabs.setCurrentIndex(tab_index)

    def close_tab(self, index):
        if index != 0:
            self.tabs.removeTab(index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # ---------------------------------------------------------
    # TRUE DARK MODE CSS STYLESHEET (Fixes Black-on-Black text)
    # ---------------------------------------------------------
    dark_style = """
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QTabWidget::pane {
            border: 1px solid #444;
            background: #2b2b2b;
        }
        QTabBar::tab {
            background: #3c3f41;
            color: #bbbbbb;
            padding: 8px 15px;
            border: 1px solid #444;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #2b2b2b;
            color: #ffffff;
            font-weight: bold;
        }
        QTableWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            gridline-color: #3f3f46;
            border: 1px solid #444;
        }
        QHeaderView::section {
            background-color: #333333;
            color: #ffffff;
            padding: 4px;
            border: 1px solid #444;
            font-weight: bold;
        }
        QPushButton {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1177bb;
        }
        QProgressBar {
            border: 1px solid #444;
            border-radius: 4px;
            text-align: center;
            color: #ffffff;
            background-color: #1e1e1e;
        }
        QProgressBar::chunk {
            background-color: #0e639c;
        }
    """
    app.setStyleSheet(dark_style)

    window = DeepfakeStudioApp()
    window.show()
    sys.exit(app.exec_())