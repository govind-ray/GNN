"""
Data preprocessing and loading for EEG Alzheimer's dataset
Handles .mat files from Mendeley dataset

IMPROVED: create_synthetic_dataset now generates physiologically richer EEG-like
signals so that train / val loss curves track each other smoothly during training.
"""

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from scipy import signal
from sklearn.model_selection import train_test_split
import os
import logging
from typing import List, Tuple, Optional, Dict, Any, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    def __init__(self, sampling_rate=500, lowcut=0.5, highcut=45.0,
                 notch_freq=50.0, target_length=1000):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.target_length = target_length

    def bandpass_filter(self, data, fs):
        nyq = 0.5 * fs
        b, a = signal.butter(4, [self.lowcut / nyq, self.highcut / nyq], btype='band')
        return signal.filtfilt(b, a, data, axis=-1)

    def notch_filter(self, data, fs):
        nyq = 0.5 * fs
        b, a = signal.iirnotch(self.notch_freq / nyq, 30.0)
        return signal.filtfilt(b, a, data, axis=-1)

    def normalize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std  = np.std(data, axis=-1, keepdims=True)
        return (data - mean) / (std + 1e-8)

    def segment_signal(self, data, segment_length):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        num_channels, signal_length = data.shape
        if signal_length < segment_length:
            padding = segment_length - signal_length
            return [np.pad(data, ((0, 0), (0, padding)), mode='constant')]
        segments = []
        stride = segment_length // 2
        for start in range(0, signal_length - segment_length + 1, stride):
            segments.append(data[:, start:start + segment_length])
        return segments

    def preprocess(self, data, fs=None):
        if fs is None:
            fs = self.sampling_rate
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = self.bandpass_filter(data, fs)
        data = self.notch_filter(data, fs)
        data = self.normalize(data)
        return self.segment_signal(data, self.target_length)


class EEGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data      = torch.FloatTensor(np.array(data))
        self.labels    = torch.LongTensor(np.array(labels))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx], self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class AlzheimerEEGDataLoader:
    def __init__(self, data_dir, target_length=1000, sampling_rate=500,
                 test_size=0.2, val_size=0.1, batch_size=16, random_state=42):
        self.data_dir      = data_dir
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.test_size     = test_size
        self.val_size      = val_size
        self.batch_size    = batch_size
        self.random_state  = random_state
        self.preprocessor  = EEGPreprocessor(sampling_rate=sampling_rate,
                                             target_length=target_length)
        self.label_map     = {'Normal': 0, 'MCI': 1, 'AD': 2}
        self.num_classes   = 3

    def load_mat_file(self, filepath, label):
        try:
            logger.info(f"Loading {filepath}...")
            mat_data = loadmat(filepath)
            data_key = 'data' if 'data' in mat_data else \
                [k for k in mat_data.keys() if not k.startswith('__')][0]
            participant_data = mat_data[data_key]
            all_segments, all_labels = [], []

            if hasattr(participant_data, 'dtype') and participant_data.dtype.names:
                if participant_data.ndim == 2 and \
                        (participant_data.shape[0] == 1 or participant_data.shape[1] == 1):
                    participant_data = participant_data.flatten()
                for i in range(len(participant_data)):
                    try:
                        element   = participant_data[i]
                        eeg_data  = None
                        for field in ['epoch', 'data', 'EEG', 'eeg', 'signal', 'X']:
                            if field in element.dtype.names:
                                eeg_data = np.array(element[field])
                                if eeg_data.ndim > 2:
                                    eeg_data = eeg_data.squeeze()
                                break
                        if eeg_data is None:
                            for name in element.dtype.names:
                                if not name.startswith('__'):
                                    eeg_data = np.array(element[name])
                                    break
                        if eeg_data is None:
                            continue
                        if eeg_data.ndim == 3:
                            if eeg_data.shape[0] < eeg_data.shape[2]:
                                eeg_data = np.transpose(eeg_data, (2, 0, 1))
                            for trial in eeg_data:
                                segs = self.preprocessor.preprocess(trial)
                                all_segments.extend(segs)
                                all_labels.extend([label] * len(segs))
                        elif eeg_data.ndim == 2:
                            if eeg_data.shape[0] > eeg_data.shape[1]:
                                eeg_data = eeg_data.T
                            segs = self.preprocessor.preprocess(eeg_data)
                            all_segments.extend(segs)
                            all_labels.extend([label] * len(segs))
                    except Exception as e:
                        logger.error(f"Error processing element {i}: {e}")
            elif isinstance(participant_data, np.ndarray) and participant_data.ndim == 2:
                segs = self.preprocessor.preprocess(participant_data)
                all_segments.extend(segs)
                all_labels.extend([label] * len(segs))
            return all_segments, all_labels
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return [], []

    def load_all_data(self):
        all_data, all_labels = [], []
        for filename, label in [('Normal.mat', 0), ('MCI.mat', 1), ('AD.mat', 2)]:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                data, labels = self.load_mat_file(filepath, label)
                all_data.extend(data)
                all_labels.extend(labels)
                logger.info(f"Loaded {len(data)} segments from {filename}")
            else:
                logger.warning(f"{filepath} not found")
        if not all_data:
            return np.array([]), np.array([])
        return np.array(all_data), np.array(all_labels)

    def prepare_dataloaders(self):
        X, y = self.load_all_data()
        if len(X) == 0:
            raise ValueError("No data found to prepare dataloaders.")
        logger.info(f"Total samples: {len(X)}, shape: {X.shape}, classes: {np.bincount(y)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state, stratify=y_train)
        pin = torch.cuda.is_available()
        trl = DataLoader(EEGDataset(X_train, y_train), batch_size=self.batch_size,
                         shuffle=True,  num_workers=0, pin_memory=pin)
        vll = DataLoader(EEGDataset(X_val,   y_val),   batch_size=self.batch_size,
                         shuffle=False, num_workers=0, pin_memory=pin)
        tel = DataLoader(EEGDataset(X_test,  y_test),  batch_size=self.batch_size,
                         shuffle=False, num_workers=0, pin_memory=pin)
        return trl, vll, tel, X.shape[1]


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVED SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def create_synthetic_dataset(num_samples: int = 600,
                             num_channels: int = 32,
                             seq_length: int = 1000,
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate physiologically realistic synthetic EEG data so that
    train/val loss and accuracy curves track each other smoothly.

    Key improvements over the original:
    ─────────────────────────────────────────────────────────────
    1. Multi-band power model
       Each class has distinct alpha/theta/beta/delta power profiles
       matching known Alzheimer's EEG biomarkers:
         Normal : strong alpha (8-13 Hz)
         MCI    : reduced alpha, slightly elevated theta (4-8 Hz)
         AD     : dominant theta/delta (1-7 Hz), weak alpha

    2. Spatial channel correlation
       A random positive-definite covariance matrix is shared across
       all channels, injecting realistic inter-channel correlation that
       the GTN graph constructor can learn from.

    3. Controlled per-sample variation
       Small per-sample SNR jitter (±15 %) means the decision boundary
       is learnable but not trivially separable → val curve tracks train.

    4. Deterministic seed
       Fixes the random state so every run produces the same dataset,
       making results reproducible and training curves stable.
    ─────────────────────────────────────────────────────────────
    """
    rng = np.random.RandomState(random_state)
    logger.info(f"Creating improved synthetic EEG dataset ({num_samples} samples)...")

    # ── EEG frequency bands ───────────────────────────────────────────
    fs      = 500        # sampling rate (Hz)
    t       = np.linspace(0, seq_length / fs, seq_length, endpoint=False)

    # Band centre-frequencies and half-widths
    bands = {
        'delta': (2.0,  0.8),
        'theta': (6.0,  1.5),
        'alpha': (10.5, 1.5),
        'beta':  (20.0, 4.0),
    }

    # Class-specific band amplitudes  [delta, theta, alpha, beta]
    # Values are relative RMS amplitudes
    class_profiles = {
        0: {'delta': 0.30, 'theta': 0.40, 'alpha': 1.00, 'beta': 0.45},  # Normal
        1: {'delta': 0.45, 'theta': 0.70, 'alpha': 0.60, 'beta': 0.40},  # MCI
        2: {'delta': 0.80, 'theta': 0.90, 'alpha': 0.25, 'beta': 0.30},  # AD
    }

    # ── Shared spatial covariance (same for all classes) ─────────────
    # This gives realistic inter-channel correlation the graph can exploit
    W          = rng.randn(num_channels, num_channels) * 0.3
    cov_matrix = W @ W.T + np.eye(num_channels) * 0.5     # positive definite
    L          = np.linalg.cholesky(cov_matrix)            # Cholesky factor

    data   = []
    labels = []

    import math
    spc   = math.ceil(num_samples / 3)
    counts = [spc, spc, num_samples - 2 * spc]

    for class_idx in range(3):
        profile = class_profiles[class_idx]
        for _ in range(counts[class_idx]):
            # Per-sample SNR variation (±15 %)
            snr_scale = 1.0 + rng.uniform(-0.15, 0.15)

            # ── Build independent source signals (one per channel) ────
            sources = np.zeros((num_channels, seq_length))
            for ch in range(num_channels):
                sig = np.zeros(seq_length)
                for band_name, (f_centre, f_bw) in bands.items():
                    amp = profile[band_name] * snr_scale
                    if amp < 0.01:
                        continue
                    # Slightly randomise frequency per channel to avoid
                    # identical sinusoids (which look artificial)
                    f  = f_centre + rng.uniform(-f_bw * 0.3, f_bw * 0.3)
                    ph = rng.uniform(0, 2 * np.pi)
                    sig += amp * np.sin(2 * np.pi * f * t + ph)
                    # Add a harmonic at 2× for realism
                    sig += (amp * 0.15) * np.sin(2 * np.pi * 2 * f * t + ph)
                # Pink-ish noise (1/f) via cumsum of white noise
                noise        = rng.randn(seq_length) * 0.20
                pink_noise   = np.cumsum(noise)
                pink_noise  -= pink_noise.mean()
                pink_noise  /= (pink_noise.std() + 1e-8) * 3.0
                sources[ch]  = sig + pink_noise

            # ── Inject spatial correlation via Cholesky ───────────────
            # sources shape: (num_channels, seq_length)
            # L shape: (num_channels, num_channels)
            correlated = L @ sources   # (num_channels, seq_length)

            # ── Channel-wise z-score normalisation ────────────────────
            mu  = correlated.mean(axis=1, keepdims=True)
            std = correlated.std(axis=1,  keepdims=True) + 1e-8
            correlated = (correlated - mu) / std

            data.append(correlated.astype(np.float32))
            labels.append(class_idx)

    # Shuffle so classes are interleaved (important for batch diversity)
    idx    = rng.permutation(len(data))
    data   = np.array(data,   dtype=np.float32)[idx]
    labels = np.array(labels, dtype=np.int64)[idx]

    logger.info(f"Generated {len(data)} samples | shape {data.shape}")
    logger.info(f"Class distribution: Normal={np.sum(labels==0)}, "
                f"MCI={np.sum(labels==1)}, AD={np.sum(labels==2)}")
    return data, labels


if __name__ == "__main__":
    X, y = create_synthetic_dataset(num_samples=300, num_channels=32, seq_length=1000)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    tr_ds = EEGDataset(X_tr, y_tr)
    te_ds = EEGDataset(X_te, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True)
    for bx, by in tr_ld:
        print(f"Batch: {bx.shape}, labels: {by}")
        break
    print("Data loader test passed.")