import sys
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim

def main():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch-directml imported successfully")
    
    if not torch_directml.is_available():
        print("FAILURE: DirectML is not available.")
        return
        
    device = torch_directml.device()
    print(f"DirectML is available. Selected device: {device}")
    
    try:
        # 1. Matrix multiplication test
        print("\n--- Testing Matrix Multiplication ---")
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)
        c = torch.matmul(a, b)
        print("Matrix multiplication SUCCESS")
        
        # 2. Neural network test
        print("\n--- Testing Neural Network Forward/Backward ---")
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        ).to(device)
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        inputs = torch.randn(4, 10, device=device)
        labels = torch.tensor([0, 1, 0, 1], device=device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print(f"Forward pass SUCCESS, loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Backward pass and optimizer step SUCCESS")
        
        print("\nOVERALL TEST: SUCCESS")
        
    except Exception as e:
        print(f"\nFAILURE: Exception occurred during tensor operations on DirectML.")
        print(e)

if __name__ == '__main__':
    main()
