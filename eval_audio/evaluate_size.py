import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from feature_cnns import AudioFeatureCNN
from feature_projections import AudioFeatureProject

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    for model in [AudioFeatureCNN(), AudioFeatureProject()]:
        total, trainable = count_parameters(model)
        print(f"Model: {model.__class__.__name__}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print()
