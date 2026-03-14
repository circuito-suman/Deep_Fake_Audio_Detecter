import argparse
import logging
import os
import sys
import yaml
from src.features.extract import FeatureExtractor
from src.models.evaluation import ModelEvaluator
from src.utils.data_processing import DataLoader, SyntheticDataGenerator
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, help='Mode: "train" or "synthetic" (overrides config)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found!")
        sys.exit(1)
        
    config = load_config(args.config)
    
    # CLI override
    if args.mode:
        config['mode'] = args.mode
    
    logger.info(f"Running in {config['mode']} mode")
    
    X, y = None, None
    
    if config['mode'] == 'synthetic':
        logger.info("Generating synthetic data for demonstration...")
        generator = SyntheticDataGenerator(config)
        X, y = generator.generate_data()
        logger.info(f"Generated {X.shape[0]} samples with {X.shape[1]} features.")
    
    elif config['mode'] == 'train':
        data_dir = config['data']['raw_dir']
        logger.info(f"Loading data from {data_dir}...")
        
        if not os.path.exists(data_dir):
            logger.error(f"Data directory {data_dir} not found!")
            sys.exit(1)
            
        extractor = FeatureExtractor(config)
        loader = DataLoader(config, extractor)
        X, y = loader.load_data()
        
        if len(X) == 0:
            logger.error("No audio files found or features extracted!")
            sys.exit(1)
            
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_split'], 
        random_state=config['random_seed']
    )
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Initialize Evaluator
    evaluator = ModelEvaluator(config)
    
    # Train Classifiers
    evaluator.train_classifiers(X_train, y_train)
    
    # Evaluate
    evaluator.evaluate_classifiers(X_test, y_test)
    
    # Plot results
    logger.info("Plotting results...")
    evaluator.plot_confusion_matrices(X_test, y_test)
    evaluator.plot_roc_curves(y_test)
    
    logger.info(f"Done! Check '{config['outputs']['plots_dir']}' directory for visualization.")

if __name__ == "__main__":
    main()
