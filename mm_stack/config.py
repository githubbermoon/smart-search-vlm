from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _default_stack_root() -> Path:
    configured = _env_path("SMART_STACK_ROOT")
    if configured is not None:
        return configured
    return Path(__file__).resolve().parents[1]


def _default_vault_root() -> Path:
    configured = _env_path("SMART_STACK_VAULT_ROOT")
    if configured is not None:
        return configured

    legacy = Path.home() / "Pranjal-Obs" / "clawd"
    if legacy.exists():
        return legacy

    return Path.home() / "Library" / "Application Support" / "SmartStack"


def _default_stack_path(*parts: str) -> Path:
    return _default_stack_root().joinpath(*parts)


def _default_vault_path(*parts: str) -> Path:
    return _default_vault_root().joinpath(*parts)


@dataclass(frozen=True)
class StackConfig:
    stack_root: Path = field(default_factory=_default_stack_root)
    vault_root: Path = field(default_factory=_default_vault_root)

    sqlite_path: Path = field(default_factory=lambda: _default_vault_path("smart_stack.db"))
    lancedb_path: Path = field(default_factory=lambda: _default_vault_path("vectors.lance"))
    text_embed_socket_path: str = os.getenv(
        "SMART_STACK_TEXT_EMBED_SOCKET",
        f"/tmp/smart_stack_text_embed_{os.getuid()}.sock",
    )

    inbox_dir: Path = field(default_factory=lambda: _default_stack_path("inbox"))
    processed_dir: Path = field(default_factory=lambda: _default_stack_path("processed"))
    failed_dir: Path = field(default_factory=lambda: _default_stack_path("failed"))
    media_dir: Path = field(default_factory=lambda: _default_vault_path("Media"))
    preprocessed_dir: Path = field(default_factory=lambda: _default_stack_path(".cache", "preprocessed"))

    clip_index_name: str = "clip_index"
    text_index_name: str = "text_index"

    schema_version: str = "mm-v1"
    clip_schema_version: str = "clip-v1"
    text_schema_version: str = "text-v1"

    clip_model_name: str = os.getenv("SMART_STACK_CLIP_MODEL", "open_clip:ViT-B-32/laion2b_s34b_b79k")
    text_model_name: str = os.getenv("SMART_STACK_TEXT_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    vlm_model_name: str = os.getenv("SMART_STACK_VLM_MODEL", "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit")

    clip_dimension: int = 512
    text_dimension: int = 768

    # Typo-aware rerank fusion controls (vector + fuzzy)
    fuzzy_alpha: float = _env_float("SMART_STACK_FUZZY_ALPHA", 0.8)
    fuzzy_beta: float = _env_float("SMART_STACK_FUZZY_BETA", 0.2)
    fuzzy_ratio_threshold: float = _env_float("SMART_STACK_FUZZY_RATIO_THRESHOLD", 0.84)
    fuzzy_min_combined_score: float = _env_float("SMART_STACK_FUZZY_MIN_COMBINED_SCORE", 0.0)
    # Query planner + attribute rerank controls
    intent_appearance_weight: float = _env_float("SMART_STACK_INTENT_APPEARANCE_WEIGHT", 0.14)
    intent_activity_weight: float = _env_float("SMART_STACK_INTENT_ACTIVITY_WEIGHT", 0.12)
    intent_presence_weight: float = _env_float("SMART_STACK_INTENT_PRESENCE_WEIGHT", 0.18)
    intent_missing_person_penalty: float = _env_float("SMART_STACK_INTENT_MISSING_PERSON_PENALTY", 0.40)
    intent_missing_clothing_penalty: float = _env_float("SMART_STACK_INTENT_MISSING_CLOTHING_PENALTY", 0.35)
    intent_semi_hard_enabled: bool = _env_bool("SMART_STACK_INTENT_SEMI_HARD_ENABLED", True)

    # Phase-3 intent-aware scoring (backward-compatible additions).
    intent_weight_retrieval: float = _env_float("SMART_STACK_INTENT_WEIGHT_RETRIEVAL", 0.60)
    intent_weight_attribute: float = _env_float(
        "SMART_STACK_INTENT_WEIGHT_ATTRIBUTE",
        _env_float("SMART_STACK_INTENT_APPEARANCE_WEIGHT", 0.20),
    )
    intent_weight_relation: float = _env_float("SMART_STACK_INTENT_WEIGHT_RELATION", 0.20)
    intent_required_entity_penalty: float = _env_float("SMART_STACK_INTENT_REQUIRED_ENTITY_PENALTY", 0.35)
    intent_activity_boost: float = _env_float(
        "SMART_STACK_INTENT_ACTIVITY_BOOST",
        _env_float("SMART_STACK_INTENT_ACTIVITY_WEIGHT", 0.12),
    )
    intent_color_boost: float = _env_float("SMART_STACK_INTENT_COLOR_BOOST", 0.20)
    intent_pattern_boost: float = _env_float("SMART_STACK_INTENT_PATTERN_BOOST", 0.20)
    intent_presence_required: bool = _env_bool("SMART_STACK_INTENT_PRESENCE_REQUIRED", True)

    # Phase-1 adaptive retrieval policy controls.
    adaptive_policy_enabled: bool = _env_bool("SMART_STACK_ADAPTIVE_POLICY_ENABLED", True)
    legacy_patches_enabled: bool = _env_bool("SMART_STACK_LEGACY_PATCHES_ENABLED", True)
    policy_base_similarity_gate: float = _env_float("SMART_STACK_POLICY_BASE_SIMILARITY_GATE", 0.60)
    policy_gate_adjustment_max: float = _env_float("SMART_STACK_POLICY_GATE_ADJUSTMENT_MAX", 0.07)
    policy_max_top_k_multiplier: float = _env_float("SMART_STACK_POLICY_MAX_TOPK_MULTIPLIER", 2.0)
    policy_confidence_fallback_threshold: float = _env_float(
        "SMART_STACK_POLICY_CONFIDENCE_FALLBACK_THRESHOLD",
        0.55,
    )

    # Verification controls (low-confidence top-k only).
    verify_enabled: bool = _env_bool("SMART_STACK_VERIFY_ENABLED", True)
    verify_low_conf_threshold: float = _env_float("SMART_STACK_VERIFY_LOW_CONF_THRESHOLD", 0.72)
    verify_top_k: int = int(os.getenv("SMART_STACK_VERIFY_TOP_K", "3"))
    text_embed_daemon_autostart: bool = _env_bool("SMART_STACK_TEXT_EMBED_DAEMON_AUTOSTART", True)
    text_embed_daemon_start_timeout_ms: int = int(
        os.getenv("SMART_STACK_TEXT_EMBED_DAEMON_START_TIMEOUT_MS", "800")
    )
    search_default_mode: str = os.getenv("SMART_STACK_SEARCH_DEFAULT_MODE", "auto").strip().lower()
    search_auto_strategy_default: str = os.getenv("SMART_STACK_SEARCH_AUTO_STRATEGY_DEFAULT", "hybrid").strip().lower()
    search_semantic_fallback_threshold: int = int(
        os.getenv("SMART_STACK_SEARCH_SEMANTIC_FALLBACK_THRESHOLD", "0")
    )
    search_hybrid_rrf_k: int = int(os.getenv("SMART_STACK_SEARCH_HYBRID_RRF_K", "60"))
    search_hybrid_weight_keyword: float = _env_float("SMART_STACK_SEARCH_HYBRID_WEIGHT_KEYWORD", 0.62)
    search_hybrid_weight_semantic: float = _env_float("SMART_STACK_SEARCH_HYBRID_WEIGHT_SEMANTIC", 0.38)
    search_hybrid_candidate_k_min: int = int(os.getenv("SMART_STACK_SEARCH_HYBRID_CANDIDATE_K_MIN", "40"))
    search_hybrid_candidate_k_max: int = int(os.getenv("SMART_STACK_SEARCH_HYBRID_CANDIDATE_K_MAX", "200"))
    search_keyword_hard_cutoff: int = int(os.getenv("SMART_STACK_SEARCH_KEYWORD_HARD_CUTOFF", "150"))
    search_cross_rerank_enabled: bool = _env_bool("SMART_STACK_SEARCH_CROSS_RERANK_ENABLED", True)
    search_cross_rerank_model: str = os.getenv(
        "SMART_STACK_SEARCH_CROSS_RERANK_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    search_cross_rerank_weight: float = _env_float("SMART_STACK_SEARCH_CROSS_RERANK_WEIGHT", 0.42)
    search_cross_rerank_k_min: int = int(os.getenv("SMART_STACK_SEARCH_CROSS_RERANK_K_MIN", "12"))
    search_cross_rerank_k_max: int = int(os.getenv("SMART_STACK_SEARCH_CROSS_RERANK_K_MAX", "40"))
    search_cross_rerank_batch_size: int = int(os.getenv("SMART_STACK_SEARCH_CROSS_RERANK_BATCH_SIZE", "16"))
    search_cross_rerank_long_query_token_cutoff: int = int(
        os.getenv("SMART_STACK_SEARCH_CROSS_RERANK_LONG_QUERY_TOKEN_CUTOFF", "8")
    )
    search_cross_rerank_high_keyword_hits: int = int(
        os.getenv("SMART_STACK_SEARCH_CROSS_RERANK_HIGH_KEYWORD_HITS", "80")
    )
    search_cross_rerank_high_keyword_cap: int = int(
        os.getenv("SMART_STACK_SEARCH_CROSS_RERANK_HIGH_KEYWORD_CAP", "20")
    )
    search_confidence_abstain_threshold: float = _env_float(
        "SMART_STACK_SEARCH_CONFIDENCE_ABSTAIN_THRESHOLD",
        0.46,
    )
    search_confidence_verify_threshold: float = _env_float(
        "SMART_STACK_SEARCH_CONFIDENCE_VERIFY_THRESHOLD",
        0.70,
    )
    search_confidence_w_top1: float = _env_float("SMART_STACK_SEARCH_CONFIDENCE_W_TOP1", 0.45)
    search_confidence_w_margin: float = _env_float("SMART_STACK_SEARCH_CONFIDENCE_W_MARGIN", 0.25)
    search_confidence_w_lexical: float = _env_float("SMART_STACK_SEARCH_CONFIDENCE_W_LEXICAL", 0.20)
    search_confidence_w_rerank: float = _env_float("SMART_STACK_SEARCH_CONFIDENCE_W_RERANK", 0.10)

    max_image_dim: int = 1024
    ingest_image_batch_size: int = _env_int("SMART_STACK_INGEST_IMAGE_BATCH_SIZE", 24)
    supported_exts: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".tiff",
        ".mp4", ".mov", ".mkv", ".webm", ".avi",
        ".mp3", ".wav", ".m4a", ".aac"
    )


IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".tiff",
)

VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
)

AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3", ".wav", ".m4a", ".aac",
)

OCR_INTENT_KEYWORDS: tuple[str, ...] = (
    "invoice",
    "receipt",
    "extract text",
    "amount",
    "convert to latex",
    "number",
    "total",
    "document",
    "bill",
)

VISUAL_INTENT_KEYWORDS: tuple[str, ...] = (
    "similar",
    "looks like",
    "style",
    "layout",
    "poster",
    "design",
    "hoarding",
    "diagram",
)
