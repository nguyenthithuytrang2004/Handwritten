#!/usr/bin/env python3
"""
Training script for HWR model
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.hwr.training import train_model


def main():
    """Train the HWR model"""
    print("🚀 Starting model training...")

    try:
        # Train model
        model, history = train_model()

        # Save model
        model_path = Path("models/model_v2.h5")
        model_path.parent.mkdir(exist_ok=True)
        model.save(model_path)
        print(f"✅ Model saved to {model_path}")

        # Print final results
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(".2f")
        print(".2f")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
