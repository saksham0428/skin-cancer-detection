import torch
checkpoint_path = "e:/Project/Saksham/skin-cancer-detection/models/skin_lesion_resnet18_baseline.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict):
    print("Checkpoint is a dictionary with keys:")
    for k in checkpoint.keys():
        print(f" - {k}")
    
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if "val_f1" in checkpoint:
        print(f"Best Validation F1: {checkpoint['val_f1']}")
    elif "best_val_f1" in checkpoint:
        print(f"Best Validation F1: {checkpoint['best_val_f1']}")
else:
    print(f"Checkpoint is of type {type(checkpoint)}")
