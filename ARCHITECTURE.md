# Architecture Guide

This document describes the internal architecture of **BlitzMate**, a Python chess engine with a modular design. It is intended for contributors who need to understand how the components fit together before making changes.

---

## Project Layout

```
chess_engine/
├── engine/              # Core engine logic (no UI dependencies)
│   ├── main.py          # Engine facade — single entry point for consumers
│   ├── analyzer.py      # Post-game move classification (Brilliant → Blunder)
│   ├── config.py        # Dataclass config loaded from config.toml
│   ├── config.toml      # Tunable parameters (search, eval, UI)
│   └── core/
│       ├── search.py            # Negamax + iterative deepening + pruning
│       ├── bitboard_evaluator.py # Bitboard-based static evaluation (primary)
│       ├── loopboard_evaluator.py # Loop-based evaluator (readable reference)
│       ├── transposition.py     # Zobrist-hashed transposition table
│       ├── board.py             # python-chess Board wrapper with history
│       └── utils.py             # Shared helpers (UCI info printing)
│
├── interface/           # Communication protocols
│   ├── uci.py           # UCI protocol handler (for GUIs like Arena, CuteChess)
│   ├── api.py           # FastAPI REST API
│   └── cli.py           # Interactive terminal (human vs engine)
│
├── gui/                 # PySide6 desktop application
│   ├── main.py          # MainWindow, tab management, entry point
│   ├── helpers.py       # EnginePool, SoundManager, shared config
│   └── widgets/         # Board, eval bar, history, timeline, etc.
│
├── tests/               # pytest test suites
│   ├── test_engine.py       # Unit tests (board, evaluator, search, TT)
│   └── test_integration.py  # End-to-end (full games, UCI, API, analyzer)
│
├── setup_assets.py      # Download opening books + Syzygy tablebases
├── requirements.txt     # Top-level dependencies
└── main.py              # Scratch / testing entry point
```

---

## Module Boundaries

The project follows a strict **layered architecture**:

```
┌──────────────┐
│   gui/       │  Presentation layer (PySide6)
├──────────────┤
│ interface/   │  Protocol layer (UCI, REST, CLI)
├──────────────┤
│   engine/    │  Core logic (zero UI/IO dependencies)
└──────────────┘
```

**Rules:**
- `engine/` never imports from `interface/` or `gui/`.
- `interface/` imports from `engine/` only.
- `gui/` imports from `engine/` only.
- Each layer has its own `requirements.txt` for isolated installs.

---

## Engine Core — How a Move Is Chosen

When `search_best_move(board)` is called, the pipeline is:

```
1. Opening Book Probe (Polyglot .bin files)
   └─ Hit? → Return book move immediately.

2. Syzygy Tablebase Probe (≤5 pieces, no castling)
   └─ Hit? → Return DTZ-optimal move immediately.

3. Iterative Deepening (depth 1 → max_depth)
   │
   ├─ Aspiration Windows (depth ≥ 4)
   │   └─ Re-search with full window on fail.
   │
   └─ Negamax + Alpha-Beta at each depth
       ├─ TT Probe (Zobrist hash lookup)
       ├─ Check Extension (+1 depth when in check)
       ├─ Reverse Futility Pruning (depth ≤ 3)
       ├─ Null-Move Pruning (adaptive R)
       ├─ Futility Pruning (quiet moves)
       ├─ Move Ordering: TT → Good Captures → Killers → History
       ├─ Late Move Reductions (table-based)
       ├─ PVS (null-window for non-PV nodes)
       └─ Quiescence Search (captures + first-ply checks)
           ├─ Stand-pat / Delta pruning
           ├─ SEE pruning (losing captures)
           └─ MVV-LVA sorted captures
```

### Key Files

| File | Responsibility |
|------|---------------|
| `search.py` | The search tree: negamax, pruning, move ordering, SEE, quiescence. |
| `bitboard_evaluator.py` | Static eval: material, PSTs, pawn structure, mobility, king safety, threats, development, rook files, knight outposts, queen activity. Uses bitboard ops. |
| `loopboard_evaluator.py` | Same eval logic implemented with loops (easier to read/debug, slower). |
| `transposition.py` | Hash table mapping Zobrist keys → `(depth, value, flag, best_move)`. Depth-preferred replacement. |

---

## Evaluation — Tapered Scoring

The evaluator computes separate **middlegame (MG)** and **endgame (EG)** scores, then blends them based on remaining material:

```
phase = sum(piece_count × phase_weight)   # 0 = endgame, 24 = opening
final = (mg × phase + eg × (24 - phase)) / 24
```

Phase weights: Knight/Bishop = 1, Rook = 2, Queen = 4.

All scores are in **centipawns** (100 cp = 1 pawn). Positive favors White; the evaluator negates for Black (negamax convention).

### Eval Components

| Component | MG Weight | EG Weight | Implementation |
|-----------|-----------|-----------|----------------|
| Material + PST | Full | Full | Per-piece lookup |
| Pawn structure | 50% | 100% | Bitwise doubled/isolated/passed |
| Bishop pair | Full | Full | Count ≥ 2 |
| Mobility | Full | Full | Safe squares (minus enemy pawn attacks) |
| King safety | Full | 15% | Shield, castling, open files, zone attackers |
| Threats | Full | Full | Pawn attacks + hanging pieces |
| Development | Scaled by phase | — | Penalty for pieces on starting squares |
| Rook files | Full | Full | Open/semi-open file bonus |
| Pawn storm | MG only | — | Advanced pawns near own king |
| Knight outposts | Full | Full | Supported + safe from enemy pawns |
| Queen activity | MG only | — | Centrality bonus, back-rank penalty |

---

## Configuration

All tunable parameters live in `engine/config.toml`, loaded at import time into `engine.config.CONFIG` (a frozen dataclass tree).

```
CONFIG
├── .search   (SearchConfig)  — depth, hash_size_mb, threads, pruning toggles
├── .eval     (EvalConfig)    — material values, PSTs, bonuses, penalties
├── .analyzer (AnalyzerConfig) — move classification thresholds
└── .ui       (UIConfig)      — engine_name, engine_author
```

To override defaults, edit `config.toml`. The dataclass defaults serve as fallback if the TOML is missing or incomplete.

---

## Transposition Table

- **Hashing:** `chess.polyglot.zobrist_hash(board)` — 64-bit Zobrist key.
- **Storage:** Python dict, index = `key % max_entries`.
- **Replacement:** Depth-preferred (deeper searches overwrite shallower).
- **Entry:** `(key, depth, value, flag, best_move)` where flag ∈ {EXACT, ALPHA, BETA}.
- **Memory:** Configurable via `hash_size_mb` (~56 bytes per entry).

---

## Opening Books & Tablebases

| Asset | Format | Path | Purpose |
|-------|--------|------|---------|
| Opening books | Polyglot `.bin` | `engine/assets/openings/` | Weighted random book moves |
| Syzygy WDL | `.rtbw` | `engine/assets/syzygy/wdl/` | Win/Draw/Loss for ≤5-piece positions |
| Syzygy DTZ | `.rtbz` | `engine/assets/syzygy/dtz/` | Distance-to-zeroing for optimal play |

Downloaded via `python setup_assets.py`. The engine works without them (falls back to search).

---

## Interface Layer

### UCI (`interface/uci.py`)

Standard UCI protocol over stdin/stdout. Supports:
- `position startpos/fen ... moves ...`
- `go depth/movetime/wtime/btime/winc/binc/infinite`
- `stop`, `quit`, `ucinewgame`, `setoption`
- Time management: allocates `time_left / estimated_moves + increment`.

### REST API (`interface/api.py`)

FastAPI app with endpoints:
- `GET /board` — current position, legal moves, game state.
- `POST /position` — set FEN.
- `POST /move` — push a UCI move.
- `POST /search` — run engine search, return best move + score.
- `POST /reset` — reset to starting position.

Thread-safe via `_board_lock`. The engine instance is shared to preserve TT across requests.

### CLI (`interface/cli.py`)

Simple interactive loop: human plays White (UCI input), engine plays Black.

---

## Analyzer

`engine/analyzer.py` classifies moves by comparing the chosen move's eval against the engine's best move:

```
delta = best_eval - chosen_eval  (from player's POV)

 delta ≤ -50  →  Brilliant
 delta ≤  50  →  Excellent
 delta ≤ 150  →  Good
 delta ≤ 300  →  Inaccuracy
 delta ≤ 600  →  Mistake
 delta >  600 →  Blunder
```

`analyze_game(moves)` iterates a full game and returns a per-move report.

---

## GUI

Built with **PySide6**. Multi-tab design where each tab is an independent game:

- `MainWindow` manages tabs, keyboard shortcuts, and the shared `EnginePool`.
- `EnginePool` runs search in a background process (avoids GIL blocking the UI).
- Each `GameTab` contains the board widget, eval bar, eval graph, move history, and captured pieces display.

The GUI imports from `engine/` but never from `interface/`.

---

## Testing

Tests use **pytest** and are split into two files:

| File | Scope | Examples |
|------|-------|---------|
| `test_engine.py` | Unit tests | Board ops, evaluator correctness, TT behavior, search tactics, move ordering, quiescence |
| `test_integration.py` | End-to-end | Full games, UCI protocol, REST API, analyzer pipeline, async search lifecycle |

Run all tests:

```bash
pytest tests/ -v
```

---

## Thread Safety

- **Search:** Runs in a daemon thread via `start_search()`. Stopped via `threading.Event` checked every 2048 nodes.
- **API:** Board access guarded by `_board_lock` (threading.Lock).
- **GUI:** Engine runs in a separate process (`EnginePool`). Qt signals bridge results back to the UI thread.
- **UCI:** Single-threaded stdin loop. Search runs in a background thread; `movetime` uses `threading.Timer` to call `stop()`.
