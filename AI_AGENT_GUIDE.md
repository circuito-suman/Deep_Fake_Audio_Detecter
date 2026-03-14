# AI Agent Guide: Deepfake Audio Detection Pipeline

## Project Overview
This project is a modular machine learning pipeline for detecting deepfake audio. It currently supports a demo mode with synthetic data and a training mode with real audio files (WAV format).

## Configuration
**Crucial:** All configurable parameters are centrally managed in `config.yaml`. 
- **DO NOT** hardcode values in python scripts.
- **ALWAYS** read from `config.yaml` when adding new features that require parameters.

### `config.yaml` Structure
- **General:** Mode (train/synthetic), random seeds.
- **Data:** Paths for raw/processed data and split ratios.
- **Features:** DSP parameters (sample rate, MFCC, Delta/Delta-Delta, Stat Moments).
- **Models:** Hyperparameters for the individual classifiers.
- **Outputs:** Directories for plots and logs.

## Codebase Structure
- `main.py`: Entry point. Loads config, orchestrates data loading, feature extraction, training, and evaluation.
- `src/features/extract.py`: `FeatureExtractor` class. Uses `librosa`, `scipy`. Calculates Temporal Statistics (Mean, Std, Skew, Kurtosis) of MFCCs + Deltas. **Dependency:** `config['features']`.
- `src/models/evaluation.py`: `ModelEvaluator` class. Wraps sklearn models. **Calculates EER (Equal Error Rate).** **Dependency:** `config['models']`.
- `src/utils/data_processing.py`: `DataLoader` and `SyntheticDataGenerator`. **Dependency:** `config['data']`, `config['synthetic']`.

## Workflow for AI Agents
1. **Check `config.yaml`** for current settings before running.
2. **Data Ingestion:** verify `data/raw/real` and `data/raw/fake` exist if running in `train` mode.
3. **Run Pipeline:** Execute `python main.py` (or `run.bat` on Windows).
4. **Analysis:** Check `plots/` for generated confusion matrices and ROC curves. Verify `logs/` for EER metrics.

## Dependencies
- `numpy`, `pandas`, `scipy`, `scikit-learn` for ML/Stats.
- `librosa`, `soundfile` for audio processing.
- `matplotlib`, `seaborn` for plotting.
- `pyyaml` for configuration management.

## Future Expansion
- When adding Deep Learning models (CNN/RNN), add a new section in `config.yaml` under `models`.
- Save trained models to `outputs.model_save_dir` (currently defined but not fully implemented).
