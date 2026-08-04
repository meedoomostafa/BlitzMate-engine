"""Engine configuration: dataclasses loaded from config.toml."""

import os
import sys
import tomllib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Legacy piece values (centipawns) for backward compatibility.
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
    depth: int = 6
    iterative_deepening: bool = True
    time_limit_ms: Optional[int] = None
    hash_size_mb: int = 64
    threads: int = 1
    use_quiescence: bool = True
    q_max_depth: int = 64
    use_null_move: bool = True
    null_move_reduction: int = 2


@dataclass
class EvalConfig:
    use_positional: bool = True

    # Material values (centipawns): middlegame.
    MATERIAL_MG: Dict[str, int] = field(
        default_factory=lambda: {
            "PAWN": 100,
            "KNIGHT": 320,
            "BISHOP": 330,
            "ROOK": 500,
            "QUEEN": 900,
            "KING": 0,
        }
    )
    # Material values (centipawns): endgame.
    MATERIAL_EG: Dict[str, int] = field(
        default_factory=lambda: {
            "PAWN": 120,
            "KNIGHT": 280,
            "BISHOP": 300,
            "ROOK": 550,
            "QUEEN": 950,
            "KING": 0,
        }
    )

    # Piece-Square Tables (PST): 64-value arrays indexed by square.
    # Middlegame knight PST.
    PST_KNIGHT_MG: List[int] = field(
        default_factory=lambda: [
            -50,
            -40,
            -30,
            -30,
            -30,
            -30,
            -40,
            -50,
            -40,
            -20,
            0,
            0,
            0,
            0,
            -20,
            -40,
            -30,
            0,
            10,
            15,
            15,
            10,
            0,
            -30,
            -30,
            5,
            15,
            20,
            20,
            15,
            5,
            -30,
            -30,
            0,
            15,
            20,
            20,
            15,
            0,
            -30,
            -30,
            5,
            10,
            15,
            15,
            10,
            5,
            -30,
            -40,
            -20,
            0,
            5,
            5,
            0,
            -20,
            -40,
            -50,
            -40,
            -30,
            -30,
            -30,
            -30,
            -40,
            -50,
        ]
    )

    # Endgame knight PST.
    PST_KNIGHT_EG: List[int] = field(
        default_factory=lambda: [
            -50,
            -40,
            -30,
            -30,
            -30,
            -30,
            -40,
            -50,
            -40,
            -20,
            0,
            5,
            5,
            0,
            -20,
            -40,
            -30,
            0,
            10,
            15,
            15,
            10,
            0,
            -30,
            -30,
            5,
            15,
            20,
            20,
            15,
            5,
            -30,
            -30,
            5,
            15,
            20,
            20,
            15,
            5,
            -30,
            -30,
            0,
            10,
            15,
            15,
            10,
            0,
            -30,
            -40,
            -20,
            0,
            5,
            5,
            0,
            -20,
            -40,
            -50,
            -40,
            -30,
            -30,
            -30,
            -30,
            -40,
            -50,
        ]
    )

    PST_KING_MG: List[int] = field(
        default_factory=lambda: [
            -10,
            10,
            15,
            -10,
            -20,
            -10,
            30,
            -10,
            -10,
            10,
            -10,
            -10,
            -10,
            -10,
            10,
            -10,
            -10,
            -20,
            -20,
            -20,
            -20,
            -20,
            -20,
            -10,
            -20,
            -30,
            -30,
            -40,
            -40,
            -30,
            -30,
            -20,
            -30,
            -40,
            -40,
            -50,
            -50,
            -40,
            -40,
            -30,
            -30,
            -40,
            -40,
            -50,
            -50,
            -40,
            -40,
            -30,
            -30,
            -40,
            -40,
            -50,
            -50,
            -40,
            -40,
            -30,
            -30,
            -40,
            -40,
            -50,
            -50,
            -40,
            -40,
            -30,
        ]
    )

    PST_KING_EG: List[int] = field(
        default_factory=lambda: [
            -50,
            -40,
            -30,
            -20,
            -20,
            -30,
            -40,
            -50,
            -30,
            -20,
            -10,
            0,
            0,
            -10,
            -20,
            -30,
            -30,
            -10,
            20,
            30,
            30,
            20,
            -10,
            -30,
            -30,
            -10,
            30,
            40,
            40,
            30,
            -10,
            -30,
            -30,
            -10,
            30,
            40,
            40,
            30,
            -10,
            -30,
            -30,
            -10,
            20,
            30,
            30,
            20,
            -10,
            -30,
            -30,
            -30,
            0,
            0,
            0,
            0,
            -30,
            -30,
            -50,
            -30,
            -30,
            -30,
            -30,
            -30,
            -30,
            -50,
        ]
    )
    PST_PAWN_MG: List[int] = field(
        default_factory=lambda: [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            10,
            10,
            20,
            30,
            30,
            20,
            10,
            10,
            5,
            5,
            10,
            25,
            25,
            10,
            5,
            5,
            0,
            0,
            0,
            20,
            20,
            0,
            0,
            0,
            5,
            -5,
            -10,
            0,
            0,
            -10,
            -5,
            5,
            5,
            10,
            10,
            -20,
            -20,
            10,
            10,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    PST_PAWN_EG: List[int] = field(
        default_factory=lambda: [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            80,
            80,
            80,
            80,
            80,
            80,
            80,
            80,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            50,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )

    # Middlegame bishop PST.
    PST_BISHOP_MG: List[int] = field(
        default_factory=lambda: [
            -20,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -20,
            -10,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -10,
            0,
            5,
            10,
            10,
            5,
            0,
            -10,
            -10,
            5,
            5,
            10,
            10,
            5,
            5,
            -10,
            -10,
            0,
            10,
            10,
            10,
            10,
            0,
            -10,
            -10,
            10,
            10,
            10,
            10,
            10,
            10,
            -10,
            -10,
            5,
            0,
            0,
            0,
            0,
            5,
            -10,
            -20,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -20,
        ]
    )

    # Middlegame rook PST.
    PST_ROOK_MG: List[int] = field(
        default_factory=lambda: [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            5,
            10,
            10,
            10,
            10,
            10,
            10,
            5,
            -5,
            0,
            0,
            0,
            0,
            0,
            0,
            -5,
            -5,
            0,
            0,
            0,
            0,
            0,
            0,
            -5,
            -5,
            0,
            0,
            0,
            0,
            0,
            0,
            -5,
            -5,
            0,
            0,
            0,
            0,
            0,
            0,
            -5,
            5,
            10,
            10,
            10,
            10,
            10,
            10,
            5,
            0,
            0,
            0,
            5,
            5,
            0,
            0,
            0,
        ]
    )

    PST_QUEEN_MG: List[int] = field(
        default_factory=lambda: [
            -20,
            -10,
            -10,
            -5,
            -5,
            -10,
            -10,
            -20,
            -10,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -10,
            0,
            5,
            5,
            5,
            5,
            0,
            -10,
            -5,
            0,
            5,
            5,
            5,
            5,
            0,
            -5,
            0,
            0,
            5,
            5,
            5,
            5,
            0,
            -5,
            -10,
            5,
            5,
            5,
            5,
            5,
            0,
            -10,
            -10,
            0,
            5,
            0,
            0,
            0,
            0,
            -10,
            -20,
            -10,
            -10,
            -5,
            -5,
            -10,
            -10,
            -20,
        ]
    )

    # Endgame bishop PST.
    PST_BISHOP_EG: List[int] = field(
        default_factory=lambda: [
            -20,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -20,
            -10,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -10,
            0,
            10,
            10,
            10,
            10,
            0,
            -10,
            -10,
            0,
            10,
            15,
            15,
            10,
            0,
            -10,
            -10,
            0,
            10,
            15,
            15,
            10,
            0,
            -10,
            -10,
            0,
            10,
            10,
            10,
            10,
            0,
            -10,
            -10,
            0,
            0,
            0,
            0,
            0,
            0,
            -10,
            -20,
            -10,
            -10,
            -10,
            -10,
            -10,
            -10,
            -20,
        ]
    )

    PST_ROOK_EG: List[int] = field(
        default_factory=lambda: [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            10,
            15,
            15,
            15,
            15,
            15,
            15,
            10,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            10,
            15,
            15,
            15,
            15,
            15,
            15,
            10,
            0,
            0,
            0,
            5,
            5,
            0,
            0,
            0,
        ]
    )

    PST_QUEEN_EG: List[int] = field(
        default_factory=lambda: [
            -20,
            -10,
            -10,
            -5,
            -5,
            -10,
            -10,
            -20,
            -10,
            0,
            5,
            5,
            5,
            5,
            0,
            -10,
            -10,
            5,
            10,
            10,
            10,
            10,
            5,
            -10,
            -5,
            5,
            10,
            15,
            15,
            10,
            5,
            -5,
            -5,
            5,
            10,
            15,
            15,
            10,
            5,
            -5,
            -10,
            5,
            10,
            10,
            10,
            10,
            5,
            -10,
            -10,
            0,
            5,
            5,
            5,
            5,
            0,
            -10,
            -20,
            -10,
            -10,
            -5,
            -5,
            -10,
            -10,
            -20,
        ]
    )

    BISHOP_PAIR_BONUS: int = 50
    ROOK_OPEN_FILE_BONUS: int = 25
    ROOK_SEMI_OPEN_FILE_BONUS: int = 15
    PASSED_PAWN_BONUS: List[int] = field(
        default_factory=lambda: [0, 10, 20, 40, 80, 150, 250, 0]
    )
    ISOLATED_PAWN_PENALTY: int = -20
    DOUBLED_PAWN_PENALTY: int = -20
    UNDEVELOPED_PENALTY: int = 25
    KING_OPEN_FILE_PENALTY: int = 25
    KING_SEMI_OPEN_FILE_PENALTY: int = 15


@dataclass
class AnalyzerConfig:
    TH_BRILLIANT: int = -50
    TH_BEST: int = 50
    TH_GOOD: int = 150
    TH_INACCURACY: int = 300
    TH_MISTAKE: int = 600


@dataclass
class UIConfig:
    engine_name: str = "BlitzMate"
    engine_author: str = "Medo"
    enable_uci: bool = True
    uci_threads: int = 1


@dataclass
class Config:
    search: SearchConfig = field(default_factory=SearchConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @staticmethod
    def load_from_toml(path: str = None) -> "Config":
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config.toml"
            )
        cfg = Config()
        if not os.path.exists(path):
            return cfg

        # Warnings go to stderr: engine.config is imported before the UCI
        # handshake, and stray stdout lines would corrupt the protocol.
        def _warn(msg: str) -> None:
            print(f"Warning: {msg}", file=sys.stderr)

        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
            # Load all config sections. Unknown keys and type mismatches are
            # rejected loudly so tuning values are never silently discarded.
            sections = {
                "search": cfg.search,
                "eval": cfg.eval,
                "analyzer": cfg.analyzer,
                "ui": cfg.ui,
            }
            for section_name, target in sections.items():
                section = raw.get(section_name, {})
                if not isinstance(section, dict):
                    _warn(
                        f"config.toml key '{section_name}' should be a "
                        f"[{section_name}] table and was ignored"
                    )
                    continue
                for k, v in section.items():
                    if not hasattr(target, k):
                        _warn(
                            f"config.toml key [{section_name}].{k} does not "
                            f"match any {type(target).__name__} attribute "
                            f"and was ignored"
                        )
                        continue
                    current = getattr(target, k)
                    # Exact type match: isinstance() would let bool slip
                    # through for int attributes (bool subclasses int).
                    if current is not None and type(v) is not type(current):
                        _warn(
                            f"config.toml key [{section_name}].{k} expects "
                            f"{type(current).__name__}, got "
                            f"{type(v).__name__}; ignored"
                        )
                        continue
                    setattr(target, k, v)
            for name, value in raw.items():
                if name not in sections:
                    kind = "section" if isinstance(value, dict) else "key"
                    _warn(
                        f"config.toml {kind} '{name}' is not recognized "
                        f"and was ignored"
                    )
        except Exception as e:
            _warn(f"Could not load config.toml: {e}")
        return cfg


CONFIG = Config.load_from_toml()
