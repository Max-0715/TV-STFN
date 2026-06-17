import torch
from dataset import TetraViewDataset, tetra_view_collate
from benchmark_tvstfn_fast import move_batch_to_device
from model import TetraViewNet
from loss import CompositeLoss
import torch.optim as optim
from torch.utils.data import DataLoader

ds = TetraViewDataset("/data/workplace/jwx/TV-STFN/tetraview_processed")
loader = DataLoader(ds, batch_size=64, collate_fn=tetra_view_collate, shuffle=True)
model = TetraViewNet().cuda()
criterion = CompositeLoss(lambda_focal=1.0, lambda_rank=0.1, lambda_mse=1.0).cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

model.train()
cnt = 0
for epoch in range(3):
    for b in loader:
        inp = move_batch_to_device(b, 'cuda')
        targets = b['targets'].cuda()
        preds = model(inp)
        loss, m, r, f = criterion(preds, targets, classification_threshold=-6.0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if cnt % 20 == 0:
            print(f"Batch {cnt} | Loss: {loss.item():.4f} (MSE:{m:.4f} Rank:{r:.4f} Focal:{f:.4f}) | Preds mean: {preds.mean().item():.4f} std: {preds.std().item():.4f}")
        cnt += 1
