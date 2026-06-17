import sys, os
import torch
import numpy as np
from dataset import TetraViewDataset, tetra_view_collate
from model import TetraViewNet
from loss import CompositeLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

def move_batch_to_device(batch, device):
    from benchmark_tvstfn_fast import move_batch_to_device as MBTD
    return MBTD(batch, device)

# Load full dataset
ds = TetraViewDataset("/data/workplace/jwx/TV-STFN/tetraview_processed")
loader = DataLoader(ds, batch_size=64, collate_fn=tetra_view_collate, shuffle=False)

model = TetraViewNet().cuda()

# Hook to extract shared_features
extracted_features = []
def hook_fn(module, input, output):
    extracted_features.append(output.detach().cpu().numpy())

# shared_head is defined; let's hook into shared_head output
hook_handle = model.shared_head.register_forward_hook(hook_fn)

def get_all_features_and_labels(mdl):
    mdl.eval()
    global extracted_features
    extracted_features = []
    labels = []
    with torch.no_grad():
        for b in tqdm(loader, desc="Extracting"):
            inp = move_batch_to_device(b, 'cuda')
            _ = mdl(inp)
            targets = b['targets'].cpu().numpy()
            labels.append(targets)
    features_np = np.concatenate(extracted_features, axis=0)
    labels_np = np.concatenate(labels, axis=0)
    # Convert logP to binary labels using classification_threshold=-6.0
    labels_bin = (labels_np <= -6.0).astype(int) 
    return features_np, labels_bin

print("Extracting features BEFORE training...")
features_before, labels_bin = get_all_features_and_labels(model)
np.save("features_before.npy", features_before)
np.save("labels.npy", labels_bin)

print("Training model for 15 epochs to get 'After' features...")
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
criterion = CompositeLoss(lambda_focal=1.0, lambda_rank=0.1, lambda_mse=1.0).cuda()
train_loader = DataLoader(ds, batch_size=32, collate_fn=tetra_view_collate, shuffle=True)

model.train()
for epoch in range(15):
    epoch_loss = 0.0
    for b in tqdm(train_loader, desc=f"Epoch {epoch+1}/15"):
        inp = move_batch_to_device(b, 'cuda')
        targets = b['targets'].cuda()
        preds = model(inp)
        loss, _, _, _ = criterion(preds, targets, classification_threshold=-6.0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader):.4f}")

print("Extracting features AFTER training...")
features_after, _ = get_all_features_and_labels(model)
np.save("features_after.npy", features_after)
print("Done extracting. Saved to features_before.npy, features_after.npy, labels.npy.")
