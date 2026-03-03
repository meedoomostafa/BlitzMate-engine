# BlitzMate Chess Engine

**A chess engine being rewritten in C++ for maximum performance, with Python bindings for the GUI and interface layers.**

---

## Current State (Python v1.0)

BlitzMate is a working chess engine written in Python. It features:

- Negamax search with alpha-beta pruning, PVS, LMR, null-move pruning
- Iterative deepening with aspiration windows
- Quiescence search with SEE pruning
- Tapered evaluation (middlegame/endgame interpolation)
- Bitboard-based pawn structure analysis
- Piece-square tables for all pieces
- Zobrist transposition table with depth-preferred replacement
- Opening book support (Polyglot .bin)
- Syzygy endgame tablebases (up to 5 pieces)
- PySide6 GUI with drag-and-drop, eval bar, move history
- UCI protocol support
- FastAPI REST interface

The Python engine searches at depth 5-6 in standard time controls. The C++ rewrite targets 10x+ speed improvement by eliminating interpreter overhead and using hardware-optimized bitboard operations.

---

## Architecture

```
chess_engine/
+-- engine_cpp/          # C++ engine (new, in development)
|   +-- src/
|   |   +-- eval.cpp
|   |   +-- eval.h
|   |   +-- search.cpp
|   |   +-- search.h
|   |   +-- tt.cpp
|   |   +-- tt.h
|   |   +-- bindings.cpp
|   +-- tests/
|   +-- CMakeLists.txt
|   +-- chess.hpp         # chess-library (header-only)
|
+-- engine_py/           # Python engine (current, reference implementation)
|   +-- core/
|   |   +-- search.py
|   |   +-- bitboard_evaluator.py
|   |   +-- transposition.py
|   |   +-- board.py
|   +-- config.py
|   +-- assets/          # Opening books + Syzygy tablebases
|
+-- interface/           # Protocol layer (Python, calls engine via bindings)
|   +-- uci.py
|   +-- api.py
|   +-- cli.py
|
+-- gui/                 # PySide6 desktop app (Python, calls engine via bindings)
|   +-- main.py
|   +-- helpers.py
|   +-- widgets/
|
+-- tests/
    +-- test_engine.py
    +-- test_integration.py
```

**Layer rules:**
- `engine_cpp/` and `engine_py/` have zero UI/IO dependencies.
- `interface/` and `gui/` import from the engine only.
- After the C++ rewrite, `interface/` and `gui/` will import from a Python binding module (`blitzmate`) instead of `engine_py.core`.

**Migration strategy:**
- The current `engine/` directory will be renamed to `engine_py/` (reference implementation).
- The new C++ engine will live in `engine_cpp/`.
- Both will coexist during development. Once the C++ engine reaches parity, `engine_py/` becomes archive/reference only.

---

## C++ Migration Plan

### Prerequisites to Study

Before starting the rewrite, study these topics in order:

**1. C++ Fundamentals**
- Modern C++ (C++17 minimum, C++20 preferred)
- RAII, smart pointers, move semantics
- Templates and constexpr
- STL containers and algorithms

**2. Build Systems**
- CMake (the standard for C++ projects)
- Compiler flags for optimization (-O3, -march=native, -flto)
- Static analysis tools (clang-tidy, cppcheck)

**3. Chess Programming**
- Bitboard representation (64-bit integers, one per piece type per color)
- Tapered evaluation (middlegame/endgame interpolation)
- Alpha-beta search with pruning techniques
- Reference: [Chess Programming Wiki](https://www.chessprogramming.org)

**4. Python Bindings**
- pybind11 (mature, widely used) or nanobind (lighter, faster compile)
- Exposing C++ classes and functions to Python
- GIL management for background search threads
- Building wheels with scikit-build-core

**5. Testing**
- Google Test or Catch2 for C++ unit tests
- Perft testing for move generation correctness
- EPD test suites for search/eval validation

**6. NNUE (for Phase 6)**
- [NNUE Probe](https://github.com/dshawul/nnue-probe) -- reference implementation
- [Stockfish NNUE docs](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md) -- training pipeline
- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) -- PyTorch trainer for Stockfish-compatible nets
- [A Guide to NNUE](https://www.chessprogramming.org/NNUE) -- Chess Programming Wiki overview
- [Bullet trainer](https://github.com/jw1912/bullet) -- fast NNUE trainer in Rust
- Training data: generate self-play games with the HCE engine, label with search scores, train HalfKAv2 architecture

### Board Representation

Using [chess-library](https://github.com/Disservin/chess-library) by Disservin:

- Header-only, single `#include "chess.hpp"`
- Full bitboard representation internally
- ~220-355M NPS in perft (Ryzen 9 5950X)
- Used by Stockfish WDL tooling, Rice engine (~3.3k Elo), fast-chess
- Provides `makeMove()` / `unmakeMove()` with incremental Zobrist hashing
- Supports Chess960, null moves, repetition detection, castling rights

This eliminates the need to implement board representation, move generation, and FEN parsing from scratch. The engine development focuses on evaluation, search, and Python bindings.

---

### Phase 1 -- Project Setup and Board Integration

**Goal**: CMake project compiling with chess-library, basic perft validation.

- [ ] Create `engine_cpp/` directory with CMake project structure
- [ ] Integrate chess-library (`chess.hpp`) as a dependency
- [ ] Write a perft test validating move generation against known positions
- [ ] Set up Google Test / Catch2 test framework
- [ ] Configure CI build (CMake + tests)
- [ ] Add compiler flags (-O3, -march=native, -flto for release)
- [ ] Verify FEN import/export through chess-library API

### Phase 2 -- Evaluation

**Goal**: Port the Python evaluator to C++ with direct bitboard access.

- [ ] Implement tapered evaluation framework (phase 0-24, MG/EG interpolation)
- [ ] Port piece-square tables (all pieces, MG and EG)
- [ ] Implement incremental material + PST scoring on make/unmake
- [ ] Port pawn structure evaluation (doubled, isolated, passed pawns via bitboards)
- [ ] Port king safety evaluation (shield, pawn storm, zone attackers, open files)
- [ ] Port mobility evaluation (safe squares per piece, exclude enemy pawn attacks)
- [ ] Port passed pawn bonuses (rank-based, rook behind passer, connected passers)
- [ ] Port threat evaluation (pawn attacks on pieces, hanging pieces)
- [ ] Port king-pawn proximity evaluation for endgames
- [ ] Add EPD test suite comparing C++ eval output against Python evaluator
- [ ] Use `std::popcount` / `std::countr_zero` from C++20 `<bit>` header

### Phase 3 -- Search

**Goal**: Port the search with all pruning techniques, single-threaded.

- [ ] Implement transposition table (Zobrist verification, depth-preferred + age replacement)
- [ ] Implement negamax with alpha-beta pruning
- [ ] Implement iterative deepening with aspiration windows
- [ ] Implement move ordering (TT move, MVV-LVA captures, killer heuristic, history heuristic)
- [ ] Implement null-move pruning (with endgame guard)
- [ ] Implement late move reductions (table-based R values)
- [ ] Implement futility pruning (reverse futility + standard, with endgame guard)
- [ ] Implement check extensions (+1 depth)
- [ ] Implement quiescence search (captures, check evasions, SEE pruning, delta pruning)
- [ ] Implement time management (allocate time from clock + increment)
- [ ] Implement repetition detection with contempt
- [ ] Add Polyglot opening book probing
- [ ] Add EPD tactical test suite (WAC, STS)
- [ ] Differential testing: compare bestmove at fixed depth against Python engine

### Phase 4 -- Python Bindings

**Goal**: Expose the C++ engine to Python so the GUI and interface layers work unchanged.

- [ ] Set up pybind11 / nanobind binding module
- [ ] Expose `Engine` class with `search()` and `evaluate()` methods
- [ ] Implement GIL release during search (so GUI stays responsive)
- [ ] Convert final bestmove from C++ move format to python-chess `Move` object
- [ ] Bridge config.toml parameters from Python to C++ engine
- [ ] Create `blitzmate` Python package (importable as `import blitzmate`)
- [ ] Update `gui/` imports from `engine_py.core` to `blitzmate`
- [ ] Update `interface/` imports from `engine_py.core` to `blitzmate`
- [ ] Package with scikit-build-core for wheel distribution
- [ ] Verify all existing Python tests pass with C++ backend

### Phase 5 -- Integration and Testing

**Goal**: Full test coverage, GUI and interface work identically to Python version.

- [ ] Run full pytest suite (`test_engine.py` + `test_integration.py`) with C++ backend
- [ ] UCI compliance test with CuteChess or Arena
- [ ] GUI smoke test (play a full game through the PySide6 interface)
- [ ] Benchmark NPS: compare C++ vs Python engine on same positions
- [ ] ELO estimation: run gauntlet against known engines via cutechess-cli
- [ ] Add Syzygy tablebase probing via Fathom library
- [ ] Set up cibuildwheel for Linux/macOS/Windows wheel builds
- [ ] Add ASan/UBSan/TSan in debug CI configuration

### Phase 6 -- Advanced (Post-MVP)

**Goal**: Push engine strength beyond HCE limits.

- [ ] Train NNUE evaluation (HalfKAv2 architecture)
- [ ] Generate training data via self-play with HCE engine
- [ ] Integrate nnue-pytorch or Bullet trainer
- [ ] Implement Lazy SMP (shared TT, independent search threads)
- [ ] Add lock-free TT for multi-threaded search
- [ ] Explore AVX2/BMI2 SIMD optimizations where beneficial
- [ ] SPSA or Texel tuning for eval weights
- [ ] Consider own opening book format or enhanced Polyglot support

---

## Installation

```bash
git clone https://github.com/meedoomostafa/BlitzMate-engine.git
cd BlitzMate-engine

python3.13 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: download opening books and tablebases
python setup_assets.py
```

## Usage

```bash
# GUI
python -m gui.main

# CLI
python -m interface.cli

# UCI
python -m interface.uci
```

## Testing

```bash
pytest tests/ -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines.

Check the [Issues](https://github.com/meedoomostafa/BlitzMate-engine/issues) page for tasks available for contribution. Each issue is self-contained with full context and acceptance criteria.

## License

See [LICENSE](LICENSE) for details.

---

**Author**: meedoomostafa
