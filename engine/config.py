# chess_engine/config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
import tomllib  # python >=3.11; if not available use toml package

# Defaults (centipawns)
PIECE_VALUES = {
    "PAWN": 100,
    "KNIGHT": 320,
    "BISHOP": 330,
    "ROOK": 500,
    "QUEEN": 900,
    "KING": 20000,
}

@dataclass
class SearchConfig:
    depth: int = 4
    iterative_deepening: bool = True
    time_limit_ms: Optional[int] = None  # None means depth-only
    hash_size_mb: int = 64
    threads: int = 1
    use_quiescence: bool = True
    q_max_depth: int = 64
    use_null_move: bool = True
    null_move_reduction: int = 2
    randomize_ties: float = 0.06  # probability to break ties randomly

@dataclass
class EvalConfig:
    piece_values: Dict[str, int] = field(default_factory=lambda: PIECE_VALUES.copy())
    use_positional: bool = True
    positional_table_path: Optional[str] = None  # file path if you store PST externally
    mobility_weights: Dict[str, float] = field(default_factory=lambda: {
        "PAWN": 5.0, "KNIGHT": 15.0, "BISHOP": 15.0, "ROOK": 8.0, "QUEEN": 10.0, "KING": 3.0
    })
    threat_attack_weights: Dict[str, int] = field(default_factory=lambda: {
        "PAWN": 25, "KNIGHT": 50, "BISHOP": 50, "ROOK": 80, "QUEEN": 140
    })
    pawn_structure_weights: Dict[str, int] = field(default_factory=lambda: {
        "doubled_penalty": 30, "isolated_penalty": 20, "passed_bonus": 50
    })
    king_safety_weights: Dict[str, float] = field(default_factory=lambda: {
        "pawn_shield_factor": 0.15, "attacker_base": 1.0, "defender_base": 0.7
    })

@dataclass
class AnalyzerConfig:
    # thresholds in centipawns (positive = loss relative to best)
    TH_BRILLIANT: int = -50   # if player's move is better than engine best by >= 50cp
    TH_BEST: int = 50
    TH_GOOD: int = 150
    TH_INACCURACY: int = 300
    TH_MISTAKE: int = 600

@dataclass
class UIConfig:
    engine_name: str = "MyEngine"
    engine_author: str = "Medo"
    enable_uci: bool = True
    uci_threads: int = 1
    uci_hash_mb: int = 64
    api_port: int = 8000

@dataclass
class Config:
    search: SearchConfig = field(default_factory=SearchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    log_level: str = "INFO"
    cache_enabled: bool = True
    cache_path: str = ".cache/tt.pickle"

    @staticmethod
    def load_from_toml(path: str = "config.toml") -> "Config":
        cfg = Config()
        if not os.path.exists(path):
            return cfg
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        # naive merge; you can improve by mapping nested dicts to dataclasses
        if "search" in raw:
            for k,v in raw["search"].items():
                if hasattr(cfg.search, k):
                    setattr(cfg.search, k, v)
        if "eval" in raw:
            for k,v in raw["eval"].items():
                if hasattr(cfg.eval, k):
                    setattr(cfg.eval, k, v)
        if "analyzer" in raw:
            for k,v in raw["analyzer"].items():
                if hasattr(cfg.analyzer, k):
                    setattr(cfg.analyzer, k, v)
        if "ui" in raw:
            for k,v in raw["ui"].items():
                if hasattr(cfg.ui, k):
                    setattr(cfg.ui, k, v)
        return cfg

# single globally importable config instance
CONFIG = Config.load_from_toml(os.environ.get("ENGINE_CONFIG_TOML", "config.toml"))
# allow env override of depth for quick debugging
try:
    override_depth = os.environ.get("ENGINE_SEARCH_DEPTH")
    if override_depth:
        CONFIG.search.depth = int(override_depth)
except Exception:
    pass
