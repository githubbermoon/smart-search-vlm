from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import torch
from PIL import Image

try:
    import whisper
except ImportError:
    whisper = None

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None

from .config import StackConfig

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None and whisper:
            logger.info("Loading Whisper model: %s", self.model_size)
            self._model = whisper.load_model(self.model_size, device=self.device)

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        """
        Transcribes audio file using Whisper.
        Returns dict with keys: text, segments, language.
        """
        if not whisper:
            logger.warning("Whisper not installed. Skipping transcription.")
            return {"text": "", "segments": []}
        
        self._load_model()
        if not self._model:
            return {"text": "", "segments": []}

        try:
            return self._model.transcribe(str(audio_path), fp16=False)
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return {"text": "", "segments": []}


class VideoProcessor:
    def __init__(self, cfg: StackConfig):
        self.cfg = cfg
        # We assume CLIP embedder is handled outside (by the caller) to avoid reloading it

    def extract_frames(self, video_path: str | Path, fps: float = 0.5) -> Iterator[tuple[float, Image.Image]]:
        """
        Extracts frames from video at given fps (default 1 frame every 2 seconds).
        Yields (timestamp, PIL.Image).
        """
        if not VideoFileClip:
            logger.warning("MoviePy not installed. Skipping video processing.")
            return

        try:
            with VideoFileClip(str(video_path)) as clip:
                duration = clip.duration
                # Step size in seconds
                step = 1.0 / fps
                
                for t in np.arange(0, duration, step):
                    frame_np = clip.get_frame(t)
                    image = Image.fromarray(frame_np)
                    
                    # Resize if too large (768px constraints)
                    w, h = image.size
                    max_dim = self.cfg.max_image_dim # 1024 or 768
                    if max(w, h) > max_dim:
                        ratio = max_dim / max(w, h)
                        new_size = (int(w * ratio), int(h * ratio))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                        
                    yield float(t), image
        except Exception as e:
            logger.error("Frame extraction failed for %s: %s", video_path, e)

    def extract_audio(self, video_path: str | Path) -> Path | None:
        """
        Extracts audio track from video to a temporary .wav file.
        Returns path to .wav file or None.
        """
        if not VideoFileClip:
            return None
            
        try:
            temp_wav = Path(video_path).with_suffix(".temp.wav")
            with VideoFileClip(str(video_path)) as clip:
                if clip.audio:
                    clip.audio.write_audiofile(str(temp_wav), logger=None)
                    return temp_wav
            return None
        except Exception as e:
            logger.error("Audio extraction failed for %s: %s", video_path, e)
            return None
