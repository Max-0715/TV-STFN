import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import random
from src.model import TetraViewNet
from src.dataset import TetraViewDataset, tetra_view_collate
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "best_tetraview_model.pth"
DATA_DIR = "dcea_dataset/processed"

def predict_sample(model_path, data_dir):
    """
    Loads a trained model and predicts on a random sample from the dataset.
    Note: Real-world inference would require full feature generation pipeline
    (RDKit -> 3D confs, Graph, etc.), which is handled during data preprocessing.
    """
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
    
    # Load Model
    print(f"Loading model from {model_path}...")
    try:
        model = TetraViewNet().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first using 'python train.py'.")
        return

    # Load Dataset
    print(f"Loading data from {data_dir}...")
    dataset = TetraViewDataset(data_dir)
    
    # Pick a random sample
    idx = random.randint(0, len(dataset)-1)
    sample_data = dataset[idx]
    
    # Collate (wrap in list to simulate batch of size 1)
    batch = tetra_view_collate([sample_data])
    
    # Prepare Input
    batch_input = {}
    batch_input['view1'] = {
        'coords': batch['view1']['coords'].to(DEVICE),
        'atom_features': batch['view1']['atom_features'].to(DEVICE),
        'num_atoms': batch['view1']['num_atoms'].to(DEVICE)
    }
    batch_input['view2'] = {
        'input_ids': batch['view2']['input_ids'].to(DEVICE),
        'attention_mask': batch['view2']['attention_mask'].to(DEVICE)
    }
    batch_input['view3'] = batch['view3'].to(DEVICE)
    batch_input['view4'] = batch['view4'].to(DEVICE)
    
    target = batch['targets'].item()
    
    # Predict
    with torch.no_grad():
        prediction = model(batch_input).item()
        
    print("\n" + "="*40)
    print(f"Prediction Result for Sample #{idx}")
    print("="*40)
    print(f"Predicted Permeability: {prediction:.4f}")
    print(f"Actual Permeability:    {target:.4f}")
    print(f"Absolute Error:         {abs(prediction - target):.4f}")
    print("="*40)

if __name__ == "__main__":
    predict_sample(MODEL_PATH, DATA_DIR)
