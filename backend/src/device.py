import logging
import torch

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Get the best available PyTorch device (CUDA, DirectML, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device
        
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            logger.info("Using DirectML device: %s", device)
            return device
    except ImportError:
        pass
        
    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device
