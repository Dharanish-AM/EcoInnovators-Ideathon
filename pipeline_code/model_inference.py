from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None





from .architecture import SimpleUNet


class ModelBundle:
    def __init__(self, pv_model: object, device: str):
        self.pv_model = pv_model
        self.device = device

    def predict_masks(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if torch is None:
            return np.zeros(image.shape[:2], dtype=np.float32), None

        tensor = self._to_tensor(image)
        with torch.inference_mode():
            pv_logits = self.pv_model(tensor)
        pv_probs = torch.sigmoid(pv_logits).squeeze(1).cpu().numpy()
        return pv_probs[0], None

    def _to_tensor(self, image: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
        arr = image.astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr)[None, ...].to(self.device)
        return tensor


def load_model(model_path: Path, device: Optional[str] = None) -> ModelBundle:
    if torch is None:
        raise ImportError("PyTorch is not installed. Please install it to use the model.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Instantiate the architecture
    model = SimpleUNet()
    
    try:
        # Try loading with weights_only=False to support custom model classes
        try:
            state = torch.load(str(model_path), map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            state = torch.load(str(model_path), map_location=device)
        
        # If it's already a model, use it directly (legacy support)
        if isinstance(state, nn.Module):
            model = state
        # If it's a state dict, load into our model
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            raise ValueError(f"Unknown model format in {model_path}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    model = model.to(device)
    return ModelBundle(pv_model=model, device=device)
