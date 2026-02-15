from __future__ import annotations

from .utils import cleanup_torch_mps


def _needs_remote_code(model_name: str) -> bool:
    return "nomic-ai/nomic-embed-text" in (model_name or "").lower()


def _prepare_text(model_name: str, text: str, is_query: bool) -> str:
    t = text.strip()
    lowered = (model_name or "").lower()
    if "nomic-embed-text" in lowered:
        return f"{'search_query' if is_query else 'search_document'}: {t}"
    if "e5" in lowered:
        return f"{'query' if is_query else 'passage'}: {t}"
    return t


class TextEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for text embeddings") from exc

        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=_needs_remote_code(self.model_name),
        )

    def unload(self) -> None:
        self.model = None
        cleanup_torch_mps()

    def __enter__(self) -> "TextEmbedder":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    def encode(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        if self.model is None:
            raise RuntimeError("Text model not loaded")

        payloads = [_prepare_text(self.model_name, text, is_query=is_query) for text in texts]
        vec = self.model.encode(payloads, normalize_embeddings=True)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [[float(x) for x in row] for row in vec]
