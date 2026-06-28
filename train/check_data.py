"""
Data Diagnostics Script for Liquid Velocity Model Training

This script helps diagnose data loading issues before training.
Run this first to verify your dataset is properly configured.
"""

import os
import sys

def check_data_files():
    """Check for data files in common locations."""
    print("="*80)
    print("DATA FILE DIAGNOSTICS")
    print("="*80)
    
    possible_paths = [
        'data/combined_annotations.csv',
        '/content/combined_annotations.csv',
        'combined_annotations.csv',
        'data/crowds_zara02_test_cleaned.txt',
        '../data/combined_annotations.csv',
    ]
    
    found_files = []
    
    for path in possible_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            found_files.append((path, size))
            print(f"✓ FOUND: {path} ({size:.2f} KB)")
        else:
            print(f"✗ NOT FOUND: {path}")
    
    print("\n" + "="*80)
    if found_files:
        print(f"Found {len(found_files)} data file(s)")
        return found_files[0][0]  # Return first found file
    else:
        print("ERROR: No data files found!")
        return None


def test_dataset_loading(data_path):
    """Test loading the dataset."""
    print("\n" + "="*80)
    print("TESTING DATASET LOADING")
    print("="*80)
    
    try:
        from datasets.pfm_trajectory_dataset_neighbours import PFM_TrajectoryDataset_neighbours
        
        print(f"Loading dataset from: {data_path}")
        dataset = PFM_TrajectoryDataset_neighbours(
            data_path,
            history_len=8,
            prediction_len=12,
            max_neighbors=4
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("\n  ERROR: Dataset is empty!")
            print("  Possible reasons:")
            print("    1. CSV file has no valid trajectories")
            print("    2. Required columns are missing")
            print("    3. Trajectories are too short (need >= 20 frames)")
            return False
        
        # Try loading one sample
        print("\n  Testing sample loading...")
        sample = dataset[0]
        print(f"  ✓ Sample loaded successfully")
        print(f"    History shape: {sample[0].shape}")
        print(f"    Future shape: {sample[1].shape}")
        print(f"    Neighbors shape: {sample[2].shape}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("  Make sure you're running from the correct directory")
        return False
    except Exception as e:
        print(f"✗ Loading Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_import():
    """Test importing the model."""
    print("\n" + "="*80)
    print("TESTING MODEL IMPORT")
    print("="*80)
    
    try:
        from models.liquid_velocity_model import LiquidVelocityModel
        print("✓ LiquidVelocityModel imported successfully")
        
        # Try creating a model
        print("\n  Testing model creation...")
        model = LiquidVelocityModel(
            hidden_size=64,
            max_neighbors=4,
            prediction_len=12
        )
        print(f"  ✓ Model created successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    Total parameters: {total_params:,}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        print("  Possible issues:")
        print("    1. 'ncps' library not installed (pip install ncps)")
        print("    2. Model file has syntax errors")
        return False
    except Exception as e:
        print(f"✗ Model Creation Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'tqdm': 'tqdm',
        'ncps': 'Neural Circuit Policies (ncps)'
    }
    
    all_good = True
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name:40s} installed")
        except ImportError:
            print(f"✗ {name:40s} NOT installed")
            print(f"  Install with: pip install {module}")
            all_good = False
    
    return all_good


def main():
    """Run all diagnostics."""
    print("\n" + "="*80)
    print("LIQUID VELOCITY MODEL - PRE-TRAINING DIAGNOSTICS")
    print("="*80 + "\n")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n" + "="*80)
        print("RESULT: Missing dependencies")
        print("="*80)
        print("Please install missing packages before training.")
        return
    
    # Check data files
    data_path = check_data_files()
    
    if data_path is None:
        print("\n" + "="*80)
        print("RESULT: No data files found")
        print("="*80)
        print("\nPlease ensure your data file is in one of these locations:")
        print("  - data/combined_annotations.csv")
        print("  - combined_annotations.csv")
        print("  - /content/combined_annotations.csv (for Colab)")
        return
    
    # Test dataset loading
    dataset_ok = test_dataset_loading(data_path)
    
    if not dataset_ok:
        print("\n" + "="*80)
        print("RESULT: Dataset loading failed")
        print("="*80)
        return
    
    # Test model import
    model_ok = test_model_import()
    
    if not model_ok:
        print("\n" + "="*80)
        print("RESULT: Model import/creation failed")
        print("="*80)
        return
    
    # All checks passed
    print("\n" + "="*80)
    print("RESULT: ALL CHECKS PASSED ✓")
    print("="*80)
    print("\nYou're ready to start training!")
    print("Run: python train/train_liquid_velocity.py")


if __name__ == "__main__":
    main()
