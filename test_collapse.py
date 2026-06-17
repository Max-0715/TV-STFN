import torch
import numpy as np
from benchmark_tvstfn_fast import move_batch_to_device, predict
from model import TetraViewNet
from dataset import TetraViewDataset, tetra_view_collate
import torch.optim as optim
from loss import CompositeLoss
from torch.utils.data import DataLoader

ds = TetraViewDataset("/data/workplace/jwx/TV-STFN/tetraview_processed")
loader = DataLoader(ds, batch_size=28, collate_fn=tetra_view_collate, shuffle=True)
model = TetraViewNet().cuda()
criterion = CompositeLoss().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(5):
    model.train()
    for i, b in enumerate(loader):
        inp = move_batch_to_device(b, 'cuda')
        targets = b['targets'].cuda()
        preds = model(inp)
        loss, _, _, _ = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i > 20: break
    
    # check eval mode stats
    model.eval()
    with torch.no_grad():
        b = next(iter(loader))
        p = model(move_batch_to_device(b, 'cuda'))
        print(f"Epoch {epoch} | train_loss: {loss.item():.3f} | eval std: {p.std().item():.5f} | mean: {p.mean().item():.5f}")

print("Done")
