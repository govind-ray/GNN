#!/usr/bin/env python3
"""
Quick Test Script - Verifies all fixes work correctly
Run this to test the debugged code without training
"""

import sys
import torch
import numpy as np

def test_imports():
    """Test that all modules can be imported"""
    print("=" * 60)
    print("TEST 1: Importing modules...")
    print("=" * 60)
    
    try:
        from hybrid_gtn_model import HybridGTN_EEG, SimpleGTN_EEG
        from data_loader import create_synthetic_dataset, EEGDataset
        from train import Trainer, PerformanceMetrics
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test that model can be instantiated"""
    print("\n" + "=" * 60)
    print("TEST 2: Creating model...")
    print("=" * 60)
    
    try:
        from hybrid_gtn_model import HybridGTN_EEG
        
        model = HybridGTN_EEG(
            num_eeg_channels=32,
            seq_length=1000,
            num_classes=3,
            feature_dim=128,
            gtn_hidden_dim=64,
            num_gtn_layers=2
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully!")
        print(f"  Total parameters: {total_params:,}")
        return True, model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\n" + "=" * 60)
    print("TEST 3: Testing forward pass...")
    print("=" * 60)
    
    try:
        # Create dummy batch
        batch_size = 4
        num_channels = 32
        seq_length = 1000
        
        x = torch.randn(batch_size, num_channels, seq_length)
        print(f"  Input shape: {x.shape}")
        
        # Forward pass
        output, embeddings = model(x)
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Check output is valid
        assert output.shape == (batch_size, 3), "Output shape mismatch!"
        assert embeddings.shape == (batch_size, num_channels, 64), "Embeddings shape mismatch!"
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_synthetic_data():
    """Test synthetic data generation"""
    print("\n" + "=" * 60)
    print("TEST 4: Testing synthetic data generation...")
    print("=" * 60)
    
    try:
        from data_loader import create_synthetic_dataset
        
        num_samples = 100
        X, y = create_synthetic_dataset(
            num_samples=num_samples,
            num_channels=32,
            seq_length=1000
        )
        
        print(f"  Requested samples : {num_samples}")
        print(f"  Generated samples : {len(X)}")
        print(f"  Data shape        : {X.shape}")
        print(f"  Labels shape      : {y.shape}")
        print(f"  Class distribution: Normal={np.sum(y==0)}, MCI={np.sum(y==1)}, AD={np.sum(y==2)}")
        
        assert len(X) == num_samples, \
            f"Sample count mismatch! Expected {num_samples}, got {len(X)}"
        assert X.shape[1] == 32,  "Channel count mismatch (expected 32)!"
        assert X.shape[2] == 1000, "Sequence length mismatch (expected 1000)!"
        assert len(np.unique(y)) == 3, "Should have exactly 3 classes!"
        
        print("✓ Synthetic data created successfully!")
        return True
    except Exception as e:
        print(f"✗ Synthetic data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_one_batch():
    """Test training for one batch"""
    print("\n" + "=" * 60)
    print("TEST 5: Testing one training batch...")
    print("=" * 60)
    
    try:
        from hybrid_gtn_model import HybridGTN_EEG
        from data_loader import create_synthetic_dataset, EEGDataset
        from torch.utils.data import DataLoader
        import torch.nn as nn
        import torch.optim as optim
        
        # Create small dataset
        X, y = create_synthetic_dataset(num_samples=16, num_channels=32, seq_length=1000)
        dataset = EEGDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        
        # Create model
        model = HybridGTN_EEG(
            num_eeg_channels=32,
            seq_length=1000,
            num_classes=3,
            feature_dim=128,
            gtn_hidden_dim=64,
            num_gtn_layers=2
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train one batch
        model.train()
        batch_x, batch_y = next(iter(loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training batch successful!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Device: {device}")
        
        return True
    except Exception as e:
        print(f"✗ Training batch failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# HYBRID GTN-EEG: DEBUGGING VERIFICATION TEST")
    print("#" * 60 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Model Creation
    success, model = test_model_creation()
    results.append(("Model Creation", success))
    
    if model is not None:
        # Test 3: Forward Pass
        results.append(("Forward Pass", test_forward_pass(model)))
    
    # Test 4: Synthetic Data
    results.append(("Synthetic Data", test_synthetic_data()))
    
    # Test 5: Training Batch
    results.append(("Training Batch", test_training_one_batch()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your code is ready to run.")
        print("\nNext steps:")
        print("  1. Quick test: python run_pipeline.py --synthetic --epochs 5")
        print("  2. Full training: python run_pipeline.py --data_dir ./data --epochs 100")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())