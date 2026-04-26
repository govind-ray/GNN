"""
run_full_evaluation.py  -  Hybrid GTN-EEG Evaluation & Visualization
=====================================================================
Always evaluates ONE model and produces ONE set of results:
  - Single split run  -> uses  best_model.pth  directly
  - CV run            -> scans all best_model_fold*.pth and picks the
                         one with the highest saved val_acc

Usage:
  python run_full_evaluation.py --checkpoint_dir checkpoints/run_YYYYMMDD_HHMMSS
"""

import torch
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse

from hybrid_gtn_model import HybridGTN_EEG
from data_loader import AlzheimerEEGDataLoader, EEGDataset, create_synthetic_dataset
from visualization import ModelVisualizer


# -----------------------------------------------------------------------------
def _find_best_model(checkpoint_dir):
    """
    Returns (label, model_path) for the single best model:
      - best_model.pth exists       -> return it directly
      - best_model_fold*.pth exist  -> return the one with highest val_acc
    """
    standard = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(standard):
        print("  Mode           : Single split")
        return '', standard

    fold_paths = sorted(glob.glob(
        os.path.join(checkpoint_dir, 'best_model_fold*.pth')))

    if not fold_paths:
        return None, None

    print("  Mode           : CV run - comparing folds to pick best one")
    best_path    = None
    best_val_acc = -1.0
    best_label   = ''

    for p in fold_paths:
        try:
            ckpt    = torch.load(p, map_location='cpu')
            val_acc = float(ckpt.get('val_acc', 0.0))
        except Exception:
            val_acc = 0.0
        label = (os.path.basename(p)
                 .replace('best_model_', '')
                 .replace('.pth', ''))
        print("    " + label + "  val_acc = " + str(round(val_acc * 100, 2)) + "%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path    = p
            best_label   = label

    print("  Best model     : " + os.path.basename(best_path) +
          "  (" + str(round(best_val_acc * 100, 2)) + "%)")
    return best_label, best_path


# -----------------------------------------------------------------------------
def _build_test_loader(config):
    data_dir      = config.get('data_dir', 'synthetic')
    use_synthetic = (
        data_dir == 'synthetic'
        or not os.path.exists(data_dir)
        or not os.listdir(data_dir)
    )
    if use_synthetic:
        print("  [INFO] No real data found - using synthetic data.")
        X, y = create_synthetic_dataset(
            num_samples  = 150,
            num_channels = config['num_channels'],
            seq_length   = config['seq_length']
        )
        return DataLoader(EEGDataset(X, y),
                          batch_size=config['batch_size'],
                          shuffle=False)

    loader = AlzheimerEEGDataLoader(
        data_dir      = data_dir,
        target_length = config['seq_length'],
        batch_size    = config['batch_size']
    )
    _, _, test_loader, _ = loader.prepare_dataloaders()
    return test_loader


# -----------------------------------------------------------------------------
def _evaluate(model, test_loader, device, checkpoint_dir, label):
    tag    = ' (Best: ' + label.replace('fold', 'Fold') + ')' if label else ''
    suffix = '_' + label if label else ''

    print()
    print('-' * 60)
    print('  Running evaluation' + tag)
    print('-' * 60)

    # Visualisations
    visualizer = ModelVisualizer(model, device)
    try:
        batch_x, batch_y = next(iter(test_loader))
        sample_data  = batch_x[0].numpy()
        sample_label = batch_y[0].item()
        print('  Generating visualisations (sample label=' +
              str(sample_label) + ')...')
        visualizer.visualize_channel_graph(
            sample_data,
            save_path=os.path.join(checkpoint_dir,
                                   'viz_channel_graph' + suffix + '.png'))
        visualizer.plot_eeg_signals(
            sample_data,
            save_path=os.path.join(checkpoint_dir,
                                   'viz_signals' + suffix + '.png'))
        visualizer.plot_feature_maps(
            sample_data,
            save_path=os.path.join(checkpoint_dir,
                                   'viz_features' + suffix + '.png'))
        visualizer.visualize_prediction(
            sample_data, true_label=sample_label,
            save_path=os.path.join(checkpoint_dir,
                                   'viz_prediction' + suffix + '.png'))
        print('  Visualisations saved.')
    except StopIteration:
        print('  [WARN] Test loader empty - skipping visualisations.')

    # Classification
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print('\n  Classification Report' + tag + ':')
    print(classification_report(all_labels, all_preds,
                                 target_names=['Normal', 'MCI', 'AD'],
                                 zero_division=0))

    # Confusion matrix
    cm     = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'MCI', 'AD'],
                yticklabels=['Normal', 'MCI', 'AD'],
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Confusion Matrix - counts' + tag, fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Normal', 'MCI', 'AD'],
                yticklabels=['Normal', 'MCI', 'AD'],
                ax=axes[1], linewidths=0.5, vmin=0, vmax=100)
    axes[1].set_title('Confusion Matrix - % per true class' + tag,
                      fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.suptitle('HybridGTN-EEG - Test Evaluation' + tag,
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir,
                             'test_confusion_matrix' + suffix + '.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  Confusion matrix saved.')


# -----------------------------------------------------------------------------
def run_evaluation(checkpoint_dir):
    print('=' * 60)
    print('  Hybrid GTN-EEG - Model Evaluation & Visualisation')
    print('=' * 60)
    print('  Checkpoint dir : ' + checkpoint_dir)

    config_path = os.path.join(checkpoint_dir, 'config.json')
    if not os.path.exists(config_path):
        print('[ERROR] config.json not found in ' + checkpoint_dir)
        return
    with open(config_path) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('  Device         : ' + str(device))

    print('\n  Scanning checkpoints...')
    label, model_path = _find_best_model(checkpoint_dir)

    if model_path is None:
        print('\n[ERROR] No model checkpoints found in:\n  ' + checkpoint_dir)
        print('\nExpected:')
        print('  best_model.pth           (single split run)')
        print('  best_model_fold*.pth     (CV run - best fold auto-selected)')
        return

    ckpt    = torch.load(model_path, map_location=device)
    val_acc = ckpt.get('val_acc', None)
    acc_str = (' (val acc: ' + str(round(val_acc * 100, 2)) + '%)') if val_acc else ''
    print('\n  Using model    : ' + os.path.basename(model_path) + acc_str)

    model = HybridGTN_EEG(
        num_eeg_channels = config['num_channels'],
        seq_length       = config['seq_length'],
        num_classes      = config['num_classes'],
        feature_dim      = config['feature_dim'],
        gtn_hidden_dim   = config['gtn_hidden_dim'],
        num_gtn_layers   = config['num_gtn_layers'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print('\n  Loading test data...')
    try:
        test_loader = _build_test_loader(config)
    except Exception as e:
        print('[ERROR] Could not build test loader: ' + str(e))
        return
    print('  Test batches   : ' + str(len(test_loader)))

    _evaluate(model, test_loader, device, checkpoint_dir, label)

    print('\n' + '=' * 60)
    print('  Evaluation complete.')
    print('  Results saved to: ' + checkpoint_dir)
    print('=' * 60)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Hybrid GTN-EEG Model - single best result',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_evaluation.py --checkpoint_dir checkpoints\\run_20240215_143022
"""
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, required=True,
        help='Path to run folder, e.g. checkpoints\\run_20240215_143022'
    )
    args = parser.parse_args()
    run_evaluation(args.checkpoint_dir) 