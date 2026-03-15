import argparse
import joblib
import yaml
import numpy as np
import os
import sys
import logging
from src.features.extract import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def predict(audio_path, model_path, config):
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train the models first using: python main.py --mode train")
        return

    # Load Model
    logger.info(f"Loading model from {model_path}...")
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Extract Features
    logger.info(f"Extracting features from {audio_path}...")
    extractor = FeatureExtractor(config)
    features = extractor.extract_features(audio_path)

    if features is None:
        logger.error("Failed to extract features from audio file.")
        return

    # Reshape for prediction (1 sample, n_features)
    features = features.reshape(1, -1)

    # Predict
    prediction = clf.predict(features)[0]
    
    # Get Probability if supported
    probability = None
    if hasattr(clf, "predict_proba"):
        probability = clf.predict_proba(features)[0]
    
    result = "FAKE" if prediction == 1 else "REAL"
    confidence = ""
    
    if probability is not None:
        prob_fake = probability[1]
        confidence = f"({prob_fake*100:.2f}% confidence it is Fake)"
        
    print("\n" + "="*40)
    print(f"RESULT: {result}")
    print(f"{confidence}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an audio file is Real or Fake Deepfake")
    parser.add_argument('input', type=str, help='Path to the input audio file (.wav, .mp3)')
    parser.add_argument('--model', type=str, default='xgboost', help='Model to use (xgboost, random_forest, svm, mlp, knn). Default: xgboost')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')

    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Map model names to filenames
    model_filename = f"{args.model}.pkl"
    # Handling specific naming conventions if any
    if args.model == 'random_forest':
        model_filename = "random_forest.pkl"
    elif args.model == 'logistic_regression':
        model_filename = "logistic_regression.pkl"
        
    model_path = os.path.join(config['outputs']['model_save_dir'], model_filename)

    predict(args.input, model_path, config)
