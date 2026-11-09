from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import image_transform


def fracture_gradcam(model, model_name: str, image: Image.Image, device: torch.device):
    """Return a 0..1 Grad-CAM map for the fracture score and the score itself."""
    model.eval()
    tensor = image_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    activations = []
    layer = model.layer4 if model_name == "resnet50" else model.features.norm5

    def capture_activation(_module, _inputs, output):
        output.retain_grad()
        activations.append(output)

    handle = layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(tensor)[0, 0]
        logit.backward()
        feature_map = activations[0]
        weights = feature_map.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, float(torch.sigmoid(logit.detach()).cpu())
    finally:
        handle.remove()


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.38):
    import matplotlib

    colors = (matplotlib.colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colors).resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), heat, alpha=alpha)
