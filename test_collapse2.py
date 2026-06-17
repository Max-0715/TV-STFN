import torch
import numpy as np
from benchmark_tvstfn_fast import move_batch_to_device
from model import TetraViewNet
from dataset import TetraViewDataset, tetra_view_collate
import torch.optim as optim
from loss import CompositeLoss
from torch.utils.data import DataLoader

device = 'cpu'

ds = TetraViewDataset("/data/workplace/jwx/TV-STFN/tetraview_processed")
loader = DataLoader(ds, batch_size=28, collate_fn=tetra_view_collate, shuffle=True)
model = TetraViewNet().to(device)
criterion = CompositeLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(1):
    model.train()
    for i, b in enumerate(loader):
        inp = move_batch_to_device(b, device)
        targets = b['targets'].to(device)
        preds = model(inp)
        loss, _, _, _ = criterion(preds, targets)
        
        if torch.isnan(loss) or not torch.isfinite(preds).all():
            print(f"FAILED AT BATCH {i}!")
            print(f"Preds finite: {torch.isfinite(preds).all().item()}")
            print(f"Loss finite: {torch.isfinite(loss).all().item()}")
            break

        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"Batch {i} | Loss: {loss.item():.3f} | gnorm: {gnorm:.3f}")
        if i > 5: break
