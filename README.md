# Deepfake Audio Detection

This project detects deepfake audio using multiple classifiers and feature extraction techniques.

## Structure

- `data/`: Raw and processed audio data.
- `src/features/`: Feature extraction scripts (MFCC, Chroma, etc.).
- `src/models/`: Classifier implementations (SVM, RF, MLP, etc.).
- `src/utils/`: Helper utilities.
- `plots/`: Generated plots (Confusion matrices, ROC curves).
- `logs/`: Execution logs.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download data (Optional):
   Populate `data/raw/real` and `data/raw/fake` with .wav files.
   
   **Recommended Datasets:**
   - **ASVspoof 2019 (LA):** [Download Link](https://datashare.ed.ac.uk/handle/10283/3336) - A standard benchmark for logical access attacks (TTS/VC).
   - **WaveFake:** [Download Link](https://zenodo.org/record/5642694) - Contains audio from multiple deepfake architectures (MelGAN, Parallel WaveGAN, etc.).
   - **In-the-Wild:** [Download Link](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) - Real-world deepfake samples.

   **Helper Script:**
   You can use the helper script to try downloading or extracting these datasets:
   ```bash
   # List available datasets
   python scripts/download_data.py --help

   # Try to download specific dataset (may require manual download for large files)
   python scripts/download_data.py --dataset asvspoof2019_la
   ```

   Or use the synthetic data generator for a demo.

## Usage

### Run with Synthetic Data (Demo)
By default, the script generates synthetic features to demonstrate the pipeline without needing large datasets.

```bash
python main.py --mode synthetic
```

### Run with Real Data
Place your audio files in `data/raw/real` and `data/raw/fake`.

```bash
python main.py --mode train --data_dir data/raw
```

## Classifiers

The system uses an ensemble of 5 classifiers:
1. Support Vector Machine (SVM)
2. Random Forest
3. Logistic Regression
4. k-Nearest Neighbors (KNN)
5. Multi-Layer Perceptron (MLP)

## Results

After running, check the `plots/` directory for:
- Confusion Matrices for each classifier.
- ROC Curves comparing performance.
