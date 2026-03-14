import argparse
import logging
import os
import sys
from src.features.extract import FeatureExtractor
from src.models.evaluation import ModelEvaluator
from src.utils.data_processing import DataLoader, SyntheticDataGenerator
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection Pipeline")
    parser.add_argument('--mode', type=str, default='synthetic', choices=['train', 'synthetic'], help='Mode: train (on real data) or synthetic (demo)')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to raw audio data')
    parser.add_argument('--model_dir', type=str, default='src/models', help='Directory to save models (not implemented for demo)')
    
    args = parser.parse_args()
    
    X, y = None, None
    
    if args.mode == 'synthetic':
        logger.info("Generating synthetic data for demonstration...")
        # Simulating 50 features (MFCC + others)
        generator = SyntheticDataGenerator(n_samples=5000, n_features=50)
        X, y = generator.generate_data()
        logger.info(f"Generated {X.shape[0]} samples with {X.shape[1]} features.")
    
    elif args.mode == 'train':
        logger.info(f"Loading data from {args.data_dir}...")
        if not os.path.exists(args.data_dir):
            logger.error(f"Data directory {args.data_dir} not found!")
            sys.exit(1)
            
        extractor = FeatureExtractor()
        loader = DataLoader(args.data_dir, extractor)
        X, y = loader.load_data()
        
        if len(X) == 0:
            logger.error("No audio files found or features extracted!")
            sys.exit(1)
            
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Initialize Evaluator
    evaluator = ModelEvaluator(output_dir='plots')
    
    # Train Classifiers
    evaluator.train_classifiers(X_train, y_train)
    
    # Evaluate
    evaluator.evaluate_classifiers(X_test, y_test)
    
    # Plot results
    logger.info("Plotting results...")
    evaluator.plot_confusion_matrices(X_test, y_test)
    evaluator.plot_roc_curves(y_test)
    
    logger.info("Done! Check 'plots/' directory for visualization.")

if __name__ == "__main__":
    main()
