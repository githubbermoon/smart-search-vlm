from __future__ import annotations

import collections
import json
import logging
import sqlite3
from typing import Any

from .config import StackConfig
from .db import connect_sqlite, ensure_schema, utc_now_iso

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, cfg: StackConfig | None = None):
        self.cfg = cfg or StackConfig()
        self.history_len = 10 # Number of past actions to track for immediate bias

    def log_activity(self, activity_type: str, details: dict[str, Any]) -> None:
        """
        Logs user activity (search, click, chat) to DB.
        """
        conn = connect_sqlite(self.cfg)
        try:
            ensure_schema(conn)
            conn.execute(
                "INSERT INTO user_activity (activity_type, details, timestamp) VALUES (?, ?, ?)",
                (activity_type, json.dumps(details), utc_now_iso()),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
        finally:
            conn.close()

    def get_recent_activity(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Retrieves recent activity to determine intent/bias.
        """
        conn = connect_sqlite(self.cfg)
        try:
            ensure_schema(conn)
            rows = conn.execute(
                "SELECT * FROM user_activity ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "type": r["activity_type"],
                    "details": json.loads(r["details"]),
                    "timestamp": r["timestamp"]
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get recent activity: {e}")
            return []
        finally:
            conn.close()

    def get_topic_bias(self) -> dict[str, float]:
        """
        Analyzes recent history to find category bias.
        Returns a dict of {category: weight_multiplier}.
        Base multiplier is 1.0. If bias found, increases to 1.2-1.5.
        """
        recent = self.get_recent_activity(self.history_len)
        categories = []
        
        # Extract categories from recent interactions
        # We assume 'details' might contain 'category' if it was a result click or chat about an image
        for act in recent:
            details = act["details"]
            if "category" in details:
                categories.append(details["category"])
            # If it was a search, we might not know category yet, unless we classified the query (future)
        
        if not categories:
            return {}

        # Count frequencies
        counts = collections.Counter(categories)
        total = len(categories)
        
        bias = {}
        for cat, count in counts.items():
            if count >= 3: # Threshold to consider it a bias
                # Simple linear boost: 1.0 + (proportion * 0.5)
                # e.g., if 5/10 are Finance, boost = 1 + 0.5 * 0.5 = 1.25
                bias[cat] = 1.0 + (count / total) * 0.5
                
        return bias

    def get_routing_adjustment(self) -> float:
        """
        Returns a shift for text_weight based on accumulated bias.
        If user is doing many deep research/finance queries (detected by category), 
        we might prefer text index (Exact match) over CLIP (vibes).
        """
        bias = self.get_topic_bias()
        # Heuristic: Finance/Academic/Technical -> Boost Text
        text_heavy = {"Finance", "Academic", "Technical"}
        
        if any(cat in bias for cat in text_heavy):
            return 0.15 # Shift hybrid weight by +0.15 towards text
        
        # Design/Personal -> Boost CLIP?
        if "Design" in bias or "Personal" in bias:
            return -0.10 # Shift towards CLIP
            
        return 0.0
