import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train

def test_resume_training(tmp_path):
    # Create a mock checkpoint dictionary
    mock_model = nn.Linear(10, 2)
    mock_optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-3)
    mock_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mock_optimizer)
    
    mock_ckpt = {
        "model_state_dict": mock_model.state_dict(),
        "optimizer_state_dict": mock_optimizer.state_dict(),
        "scheduler_state_dict": mock_scheduler.state_dict(),
        "epoch": 15,
        "val_f1": 0.755174,
        "best_val_loss": 0.558754,
        "patience_counter": 2,
    }

    # Patch dependencies
    with patch("src.train.torch.load", return_value=mock_ckpt) as mock_load, \
         patch("src.train.torch.save") as mock_save, \
         patch("src.train.train_one_epoch", return_value=(0.5, 0.8)) as mock_train_epoch, \
         patch("src.train.validate", return_value=(0.4, 0.85, 0.76)) as mock_validate, \
         patch("src.train.MODELS_DIR", tmp_path):
        
        # We need a mock loader
        mock_loader = MagicMock()
        mock_loader.__len__.return_value = 1
        
        config = {
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "num_epochs": 17,  # Max epoch = 17, resume from 15, should run 16, 17
            "model_type": "test_model",
            "checkpoint_name": "test_ckpt.pth",
            "early_stopping_patience": 5,
        }
        
        class_weights = torch.tensor([1.0, 1.0])
        device = torch.device("cpu")
        
        history = train(
            model=mock_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            class_weights=class_weights,
            config=config,
            device=device,
            resume_checkpoint="dummy_path.pth"
        )
        
        # Verify checkpoint loads
        mock_load.assert_called_once_with("dummy_path.pth", map_location="cpu", weights_only=False)
        
        # Verify start_epoch + 1 (should run epoch 16 and 17)
        assert mock_train_epoch.call_count == 2
        
        # Verify optimizer and scheduler states were actually loaded (implicitly by running fine, but we can check if they updated)
        # We can verify that the returned history has 2 entries
        assert len(history["train_loss"]) == 2
        
        # Check that it started at epoch 16 and went to 17. The history doesn't record epoch number directly,
        # but the call count for train_one_epoch is 2, which matches range(16, 17 + 1)
        
        # Verify patience state (it should start at 2). Since validate returns 0.76 > 0.755174 in the first step,
        # patience drops to 0. It shouldn't trigger early stopping.
        
        # Verify torch.save was called because val_f1 (0.76) > best_val_f1 (0.755174)
        assert mock_save.call_count > 0
        
        # Let's check the args to torch.save for patience and epoch
        saved_dict = mock_save.call_args[0][0]
        assert saved_dict["epoch"] == 16  # first epoch run
        assert saved_dict["val_f1"] == 0.76
        assert saved_dict["patience_counter"] == 0

        # Now test early stopping from resumed patience.
        # If patience_counter = 4, patience = 5, and validate returns worse.
        mock_train_epoch.reset_mock()
        mock_validate.reset_mock()
        mock_ckpt["patience_counter"] = 4
        mock_ckpt["val_f1"] = 0.99  # very high, won't beat it
        mock_validate.return_value = (1.0, 0.5, 0.5)
        
        history2 = train(
            model=mock_model,
            train_loader=mock_loader,
            val_loader=mock_loader,
            class_weights=class_weights,
            config=config,
            device=device,
            resume_checkpoint="dummy_path.pth"
        )
        # Should stop after 1 epoch (epoch 16) because patience hits 5
        assert mock_train_epoch.call_count == 1
