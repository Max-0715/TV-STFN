import torch
import os
try:
    data = torch.load('/data/workplace/jwx/TV-STFN/tetraview_processed/data_0.pt', weights_only=False)
    print("Keys:", data.keys())
    print("Target:", data['target'])
    print("View1:", type(data['view1_3d']))
except Exception as e:
    print(e)
