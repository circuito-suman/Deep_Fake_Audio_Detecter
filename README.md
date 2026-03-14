# Deepfake Audio Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A robust, modular machine learning pipeline for detecting deepfake audio (AI-generated speech). This project leverages advanced feature extraction (MFCCs, Deltas, Statistical Moments) and ensemble classification to distinguish between real and synthetic audio.

**Compatible with Windows, macOS, and Linux.**

## 🚀 Key Features

*   **Multi-Classifier Ensemble:** Utilizes SVM, Random Forest, Logistic Regression, KNN, and MLP for robust detection.
*   **Advanced Feature Extraction:**
    *   **MFCCs:** Mel-frequency cepstral coefficients.
    *   **Temporal Dynamics:** Delta and Delta-Delta coefficients to capture speech motion.
    *   **Statistical Analysis:** Mean, Variance, Skewness, and Kurtosis of features.
*   **Biometric Metrics:** Calculates **Equal Error Rate (EER)**, the standard metric for spoofing detection.
*   **Configuration Driven:** All parameters manageable via `config.yaml`.
*   **Cross-Platform:** standardized path handling for all OS environments.

## 📂 Project Structure

```
Deep_Fake_Audio_Detecter/
├── config.yaml           # Central configuration (Paramters, Paths, Models)
├── main.py               # Main entry point for training/demo
├── run.sh                # Helper script for Mac/Linux
├── run.bat               # Helper script for Windows
├── requirements.txt      # Python dependencies
├── AI_AGENT_GUIDE.md     # Technical guide for AI assistance
├── src/
│   ├── features/         # Feature extraction logic (Librosa, Scipy)
│   ├── models/           # Classifier definitions & Evaluation (EER, ROC)
│   └── utils/            # Data loading & Synthetic generation
├── data/                 # Data storage (git-ignored)
│   ├── raw/              # Place .wav files here ({real, fake})
│   └── processed/        # (Optional) Saved features
├── plots/                # Generated Confusion Matrices & ROC Curves
└── logs/                 # Execution logs
```

## 🛠️ Setup

### Prerequisites
*   Python 3.8 or higher
*   pip (Python Package Installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Deep_Fake_Audio_Detecter
    ```

2.  **Create a Virtual Environment (Recommended):**
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```cmd
        python -m venv venv
        venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data Preparation

You have two options: **Synthetic Demo** (no download needed) or **Real Data Training**.

### Option A: Synthetic Demo
Run the project immediately without downloading gigabytes of audio. The system generates statistical features simulating real vs. fake distibutions.

### Option B: Real Datasets
Download one of the following and extract them into `data/raw/`:

*   **ASVspoof 2019 (LA):** [Download](https://datashare.ed.ac.uk/handle/10283/3336) (Standard Benchmark)
*   **WaveFake:** [Download](https://zenodo.org/record/5642694) (Modern Architectures)

**Directory Structure for Real Data:**
```
data/
  raw/
    real/
      file1.wav
      file2.wav
    fake/
      deepfake1.wav
      deepfake2.wav
```

## 🏃 Usage

### Quick Start (Demo)

*   **macOS/Linux:**
    ```bash
    ./run.sh
    ```
*   **Windows:**
    Double-click `run.bat` or run:
    ```cmd
    run.bat
    ```

### Training on Real Data

1.  Ensure your data is in `data/raw/real` and `data/raw/fake`.
2.  Run the training pipeline:
    ```bash
    python main.py --mode train
    ```
3.  (Optional) Custom Configuration:
    *   Edit **`config.yaml`** to change model hyperparameters, feature extraction settings (e.g., `n_mfcc`), or paths.

## 📈 Results & Evaluation

After execution, check the `plots/` directory for:
1.  **Confusion Matrices:** Visual breakdown of True Positives, False Positives, etc.
2.  **ROC Curves:** Performance trade-off between Sensitivity and Specificity.
3.  **Console Output:** Classification Report (Precision, Recall, F1-Score) and **EER (Equal Error Rate)**.

## 🤝 Contributing

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
