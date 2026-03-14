import librosa
import numpy as np
import logging

class FeatureExtractor:
    def __init__(self, config):
        self.sample_rate = config['features']['sample_rate']
        self.n_mfcc = config['features']['n_mfcc']
        self.logger = logging.getLogger(__name__)

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            
            # Calculate mean of features (or other statistics if needed)
            features = np.hstack([
                np.mean(mfccs, axis=1),
                np.mean(chroma, axis=1),
                np.mean(mel, axis=1),
                np.mean(contrast, axis=1),
                np.mean(tonnetz, axis=1)
            ])
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features from {file_path}: {e}")
            return None
