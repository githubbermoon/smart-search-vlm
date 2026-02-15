from __future__ import annotations

from pathlib import Path

from .utils import cleanup_torch_mps


def parse_openclip_model(model_ref: str) -> tuple[str, str]:
    raw = model_ref.replace("open_clip:", "", 1)
    if "/" not in raw:
        raise ValueError("CLIP model must be in form 'open_clip:MODEL/PRETRAINED'")
    model_name, pretrained = raw.split("/", 1)
    return model_name, pretrained


class OpenCLIPEmbedder:
    def __init__(self, model_ref: str):
        self.model_ref = model_ref
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cpu"

    def load(self) -> None:
        try:
            import open_clip
            import torch
        except Exception as exc:
            raise RuntimeError("open_clip_torch is required. Install with: uv pip install open_clip_torch") from exc

        model_name, pretrained = parse_openclip_model(self.model_ref)
        self.device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(self.device)
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def unload(self) -> None:
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        cleanup_torch_mps()

    def __enter__(self) -> "OpenCLIPEmbedder":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def encode_images(self, image_paths: list[Path]) -> list[list[float]]:
        if self.model is None or self.preprocess is None:
            raise RuntimeError("CLIP model not loaded")
        import torch
        from PIL import Image

        tensors = []
        for path in image_paths:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensors.append(self.preprocess(img))
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            vec = self.model.encode_image(batch)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.detach().cpu().tolist()

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CLIP model not loaded")
        import torch

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            vec = self.model.encode_text(tokens)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.detach().cpu().tolist()
