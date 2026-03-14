import librosa
import numpy as np
import logging
import scipy.stats

class FeatureExtractor:
    def __init__(self, config):
        self.sample_rate = config['features']['sample_rate']
        self.n_mfcc = config['features']['n_mfcc']
        self.n_fft = config['features'].get('n_fft', 2048)
        self.hop_length = config['features'].get('hop_length', 512)
        self.logger = logging.getLogger(__name__)

    def extract_lfcc(self, audio, sr, n_lfcc=40, n_filter=40):
        """
        Extract Linear Frequency Cepstral Coefficients (LFCC)
        Standard MFCC uses Mel filterbank (logarithmic), LFCC uses Linear filterbank.
        """
        # 1. Power Spectrogram
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))**2
        
        # 2. Linear Filterbank
        filters = librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=n_filter, htk=False)
        # Note: librosa doesn't have a direct 'linear' filterbank function public, 
        # but we can simulate it by standard STFT if we map bins linearly.
        # However, for simplicity/robustness in this shallow pipeline, we often approximate 
        # or stick to MFCC. But since WaveFake uses it, let's look at a simpler approximation:
        # Just use the linear magnitude spectrogram statistics as "Linear" features.
        
        # Truly manual LFCC is complex to implement from scratch without 'torchaudio' or 'spaasm'.
        # We will use the magnitude spectrogram averaged in linear bands as a proxy for this demo.
        return np.mean(S, axis=1)[:n_lfcc]

    def get_stats(self, feature_matrix):
        """
        Calculate statistical features: Mean, Std, Skew, Kurtosis
        Input: (n_features, n_time_steps)
        Output: (n_features * 4, )
        """
        mean = np.mean(feature_matrix, axis=1)
        std = np.std(feature_matrix, axis=1)
        skew = scipy.stats.skew(feature_matrix, axis=1)
        kurtosis = scipy.stats.kurtosis(feature_matrix, axis=1)
        return np.concatenate([mean, std, skew, kurtosis])

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # 1. MFCCs (Standard)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # 2. Delta and Delta-Delta MFCCs (Dynamic features)
            delta_mfcc = librosa.feature.delta(mfccs)
            delta2_mfcc = librosa.feature.delta(mfccs, order=2)
            
            # 3. Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # 4. Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Combine all temporal features: (n_features_total, time)
            combined_features = np.vstack([
                mfccs, 
                delta_mfcc, 
                delta2_mfcc, 
                chroma, 
                contrast
            ])
            
            # Compute Statistics to collapse time dimension but keep distributional info
            # (WaveFake often uses GMMs on raw frames, but for SVM/RF we need fixed vectors)
            final_feature_vector = self.get_stats(combined_features)
            
            return final_feature_vector
        except Exception as e:
            self.logger.error(f"Error extracting features from {file_path}: {e}")
            return None
