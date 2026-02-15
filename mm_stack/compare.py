from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import StackConfig
from .search_engine import MultimodalSearchEngine
from .utils import cleanup_torch_mps

try:
    from PIL import Image
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

@dataclass
class ComparisonResponse:
    image_A: dict[str, Any]
    image_B: dict[str, Any]
    differences: str
    similarities: str
    conclusion: str
    confidence: str

class Comparator:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.search_engine = MultimodalSearchEngine(self.cfg)
        self.max_images = 2  # Strict limit for comparison to avoid RAM explosion
        self.max_image_size = 768

    def _resize_image(self, image_path: str) -> str:
        if not Image: return image_path
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if max(w, h) <= self.max_image_size:
                    return image_path
                
                ratio = self.max_image_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                
                temp_dir = Path(self.cfg.sqlite_path).parent / "temp_resized"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / Path(image_path).name
                
                if not temp_path.exists():
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    img.save(temp_path)
                return str(temp_path)
        except Exception:
            return image_path

    def compare(self, query: str) -> ComparisonResponse | dict[str, Any]:
        """
        Retrieves top images and compares them.
        """
        # 1. Retrieve
        # We search with top_k=3 but only take top 2 for direct comparison (A vs B)
        # to keep prompt clean.
        search_resp = self.search_engine.search(query=query, top_k=self.max_images)
        results = search_resp.results
        
        if len(results) < 2:
            return {
                "error": "Need at least 2 relevant images to compare.",
                "found": len(results)
            }

        image_paths = []
        meta_context = []
        
        for i, r in enumerate(results):
            path = self._resize_image(r["file_path"])
            image_paths.append(path)
            meta_context.append(f"Image {chr(65+i)}: {r['caption']} (Score: {r['score']:.2f})")

        # 2. Load VLM
        try:
            model, processor = load(self.cfg.vlm_model_name)
        except Exception as e:
            return {"error": f"Failed to load VLM: {e}"}

        try:
            # 3. Prompt
            system_prompt = (
                "You are a precise visual comparison engine.\n"
                "Compare Image A and Image B based on the user query.\n"
                "Return VALID JSON ONLY with this schema:\n"
                "{\n"
                '  "image_A": {"summary": "...", "key_elements": [...]},\n'
                '  "image_B": {"summary": "...", "key_elements": [...]},\n'
                '  "differences": "...",\n'
                '  "similarities": "...",\n'
                '  "conclusion": "..."\n'
                "}\n"
            )
            
            user_msg = f"Query: {query}\nContext:\n" + "\n".join(meta_context)
            
            formatted_prompt = apply_chat_template(
                processor, model.config, user_msg, num_images=len(image_paths)
            )
            
            loaded_images = [load_image(p) for p in image_paths]
            
            output = generate(
                model, 
                processor, 
                image=loaded_images, 
                prompt=formatted_prompt, 
                max_tokens=512, 
                verbose=False
            )
            
            text = output.text if hasattr(output, "text") else str(output)
            
            # Parse JSON
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    json_str = text[start:end+1]
                    data = json.loads(json_str)
                    return ComparisonResponse(
                        image_A=data.get("image_A", {}),
                        image_B=data.get("image_B", {}),
                        differences=data.get("differences", ""),
                        similarities=data.get("similarities", ""),
                        conclusion=data.get("conclusion", ""),
                        confidence="High"
                    )
            except Exception:
                pass
                
            return {
                "raw_output": text,
                "error": "Failed to parse structured comparison.",
                "image_A_meta": results[0],
                "image_B_meta": results[1]
            }

        finally:
            del model
            del processor
            cleanup_torch_mps()
            gc.collect()
