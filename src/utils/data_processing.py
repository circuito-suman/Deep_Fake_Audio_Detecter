import os
import numpy as np
import pandas as pd
from src.features.extract import FeatureExtractor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

class DataLoader:
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Loads data from data_dir. Assumes structure:
        data_dir/
            train/
                real/
                fake/
            test/
                ...
        """
        X = []
        y = []
        
        for split in ['train', 'test']:
            split_dir = os.path.join(self.data_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            for label, class_name in enumerate(['real', 'fake']):
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                    
                for file_name in tqdm(os.listdir(class_dir), desc=f"Loading {split}/{class_name}"):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(class_dir, file_name)
                        features = self.feature_extractor.extract_features(file_path)
                        if features is not None:
                            X.append(features)
                            y.append(label)
                            
        return np.array(X), np.array(y)

class SyntheticDataGenerator:
    def __init__(self, n_samples=1000, n_features=20):
        self.n_samples = n_samples
        self.n_features = n_features

    def generate_data(self):
        # Generate synthetic features for demonstration
        # Real audio: Class 0 (Gaussian centered at 0)
        X_real = np.random.normal(loc=0.0, scale=1.0, size=(self.n_samples // 2, self.n_features))
        y_real = np.zeros(self.n_samples // 2)
        
        # Fake audio: Class 1 (Gaussian centered at 2)
        X_fake = np.random.normal(loc=2.0, scale=1.5, size=(self.n_samples // 2, self.n_features))
        y_fake = np.ones(self.n_samples // 2)
        
        X = np.vstack([X_real, X_fake])
        y = np.hstack([y_real, y_fake])
        
        # Shuffle
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        return X[indices], y[indices]
