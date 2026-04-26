"""
train.py - Fixed Hybrid GTN-EEG Training Script
Fixed issues:
  1. Complete training loop implementation
  2. Proper GTN layer initialization
  3. Realistic graph axes (loss: 0-1, accuracy: 0-100%)
  4. Config path handling
  5. FIXED: Model saving logic (best_val_acc check now runs BEFORE metrics.update)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, List
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from hybrid_gtn_model import HybridGTN_EEG
from data_loader import AlzheimerEEGDataLoader, create_synthetic_dataset, EEGDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Track training metrics"""
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def update(self, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.train_losses)

    def plot_training_history(self, save_path: str):
        """Plot training history with realistic axes and smooth curves"""
        from scipy.interpolate import make_interp_spline

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = np.arange(1, len(self.train_losses) + 1)

        # -------------------------------------------------------
        # Plot 1: Loss (0-1 scale, realistic for CE loss) - SMOOTH
        # -------------------------------------------------------
        if len(epochs) > 3:
            spl_train = make_interp_spline(epochs, self.train_losses, k=3)
            spl_val   = make_interp_spline(epochs, self.val_losses,   k=3)
            x_smooth  = np.linspace(epochs[0], epochs[-1], 300)
            y_train_smooth = spl_train(x_smooth)
            y_val_smooth   = spl_val(x_smooth)

            ax1.plot(x_smooth, y_train_smooth, '-', label='Train Loss',
                     linewidth=2.5, color='#2E86AB')
            ax1.plot(x_smooth, y_val_smooth,   '-', label='Val Loss',
                     linewidth=2.5, color='#A23B72')
            ax1.plot(epochs, self.train_losses, 'o', markersize=5, color='#2E86AB', alpha=0.6)
            ax1.plot(epochs, self.val_losses,   's', markersize=5, color='#A23B72', alpha=0.6)
        else:
            ax1.plot(epochs, self.train_losses, 'o-', label='Train Loss',
                     linewidth=2.5, markersize=6, color='#2E86AB')
            ax1.plot(epochs, self.val_losses,   's-', label='Val Loss',
                     linewidth=2.5, markersize=6, color='#A23B72')

        ax1.set_ylim(0, 1.0)
        ax1.set_xlim(0, len(epochs) + 1)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss (CrossEntropyLoss)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss History', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--')

        best_idx = self.best_epoch - 1
        if 0 <= best_idx < len(self.val_losses):
            ax1.axvline(x=self.best_epoch, color='red', linestyle='--',
                        alpha=0.5, linewidth=1.5)

        # -------------------------------------------------------
        # Plot 2: Accuracy (20-105% scale) - SMOOTH
        # -------------------------------------------------------
        train_accs_pct = np.array([acc * 100 for acc in self.train_accs])
        val_accs_pct   = np.array([acc * 100 for acc in self.val_accs])

        if len(epochs) > 3:
            spl_train_acc = make_interp_spline(epochs, train_accs_pct, k=3)
            spl_val_acc   = make_interp_spline(epochs, val_accs_pct,   k=3)
            x_smooth = np.linspace(epochs[0], epochs[-1], 300)
            y_train_acc_smooth = spl_train_acc(x_smooth)
            y_val_acc_smooth   = spl_val_acc(x_smooth)

            ax2.plot(x_smooth, y_train_acc_smooth, '-', label='Train Accuracy',
                     linewidth=2.5, color='#2E86AB')
            ax2.plot(x_smooth, y_val_acc_smooth,   '-', label='Val Accuracy',
                     linewidth=2.5, color='#A23B72')
            ax2.plot(epochs, train_accs_pct, 'o', markersize=5, color='#2E86AB', alpha=0.6)
            ax2.plot(epochs, val_accs_pct,   's', markersize=5, color='#A23B72', alpha=0.6)
        else:
            ax2.plot(epochs, train_accs_pct, 'o-', label='Train Accuracy',
                     linewidth=2.5, markersize=6, color='#2E86AB')
            ax2.plot(epochs, val_accs_pct,   's-', label='Val Accuracy',
                     linewidth=2.5, markersize=6, color='#A23B72')

        ax2.set_ylim(20, 105)
        ax2.set_xlim(0, len(epochs) + 1)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Training Accuracy History', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle='--')

        if 0 <= best_idx < len(val_accs_pct):
            best_acc = val_accs_pct[best_idx]
            ax2.plot(self.best_epoch, best_acc, 'r*', markersize=20,
                     label=f'Best: {best_acc:.1f}%', zorder=5)
            ax2.axvline(x=self.best_epoch, color='red', linestyle='--',
                        alpha=0.5, linewidth=1.5)

        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

        plt.suptitle('Hybrid GTN-EEG Training Progress',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
        plt.close()


class Trainer:
    """Training wrapper for Hybrid GTN-EEG model"""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: str = 'cuda',
                 lr: float = 0.001, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.metrics = PerformanceMetrics()

    def train_epoch(self) -> Tuple[float, float]:
        """Complete training loop implementation"""
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            for batch_x, batch_y in pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def train(self, num_epochs: int, checkpoint_dir: str) -> str:
        """Complete training loop with FIXED model saving logic"""
        logger.info(f"Starting training on {self.device}")
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc     = self.validate()

            # ✅ FIX: Check BEFORE calling metrics.update(), because
            # update() itself sets best_val_acc = val_acc when it improves.
            # If we check after, val_acc == best_val_acc always → model never saves.
            is_best = val_acc > self.metrics.best_val_acc

            self.metrics.update(train_loss, train_acc, val_loss, val_acc)
            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch [{epoch:3d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                + (" *** BEST ***" if is_best else "")
            )

            if is_best:
                logger.info(f"  >>> New best model! Val acc: {val_acc:.4f} — saving checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss':   val_loss,
                    'train_acc':  train_acc,
                    'val_acc':    val_acc,
                }, best_model_path)

        if not os.path.exists(best_model_path):
            logger.warning("No best model was saved during training! "
                           "Saving final epoch model as fallback...")
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss':   val_loss,
                'train_acc':  train_acc,
                'val_acc':    val_acc,
            }, best_model_path)

        return best_model_path


def main(args: Any) -> str:
    """Main training function"""

    # -------------------------------------------------------
    # Setup
    # -------------------------------------------------------
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------
    use_synthetic = args.synthetic or not os.path.exists(args.data_dir)

    if use_synthetic:
        logger.info("Creating synthetic EEG dataset...")
        X, y = create_synthetic_dataset(
            num_samples=600,
            num_channels=32,
            seq_length=1000
        )

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1/0.8, random_state=42, stratify=y_train)

        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            EEGDataset(X_train, y_train),
            batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=pin
        )
        val_loader = DataLoader(
            EEGDataset(X_val, y_val),
            batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=pin
        )

        num_channels  = 32
        data_source   = 'synthetic'
    else:
        logger.info(f"Loading real data from {args.data_dir}...")
        loader = AlzheimerEEGDataLoader(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        train_loader, val_loader, _, num_channels = loader.prepare_dataloaders()
        data_source = os.path.abspath(args.data_dir)

    # -------------------------------------------------------
    # Model Setup
    # -------------------------------------------------------
    logger.info("Creating Hybrid GTN-EEG model...")
    model = HybridGTN_EEG(
        num_eeg_channels=num_channels,
        seq_length=1000,
        num_classes=3,
        feature_dim=128,
        gtn_hidden_dim=64,
        num_gtn_layers=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        lr=args.lr
    )

    best_model_path = trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir
    )

    # -------------------------------------------------------
    # Save Config
    # -------------------------------------------------------
    config = {
        'num_channels':   num_channels,
        'seq_length':     1000,
        'num_classes':    3,
        'feature_dim':    128,
        'gtn_hidden_dim': 64,
        'num_gtn_layers': 2,
        'batch_size':     args.batch_size,
        'learning_rate':  args.lr,
        'epochs':         args.epochs,
        'data_dir':       data_source,
        'device':         str(device),
        'timestamp':      timestamp,
        'best_val_acc':   trainer.metrics.best_val_acc,
        'best_epoch':     trainer.metrics.best_epoch,
    }

    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # -------------------------------------------------------
    # Plot Training History
    # -------------------------------------------------------
    history_path = os.path.join(checkpoint_dir, 'training_history.png')
    trainer.metrics.plot_training_history(history_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete!")
    logger.info(f"Best validation accuracy: {trainer.metrics.best_val_acc:.4f}")
    logger.info(f"Best epoch: {trainer.metrics.best_epoch}")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"{'='*60}\n")

    return checkpoint_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Hybrid GTN-EEG Model')
    parser.add_argument('--data_dir',   type=str,   default='./data',
                        help='Path to data directory')
    parser.add_argument('--synthetic',  action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--epochs',     type=int,   default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int,   default=16,
                        help='Batch size')
    parser.add_argument('--lr',         type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str,   default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--no_cv',      action='store_true',
                        help='Skip cross-validation')
    parser.add_argument('--n_folds',    type=int,   default=5,
                        help='Number of folds for CV')

    args = parser.parse_args()
    checkpoint_dir = main(args)
    print(f"\nResults saved to: {checkpoint_dir}")