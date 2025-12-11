# ♟️ BlitzMate Chess Engine

**Version 1.0 (Stable)**

> **A strategic, Python-based chess engine featuring a custom Negamax search, Tapered Evaluation, and a modern GUI.**

-----

## Project Overview

BlitzMate has evolved from a basic move calculator into a strategic engine. It moves beyond simple material counting to understand positional nuances like piece activity, king safety, and pawn structures.

The project is structured into three independent modules:

  * **Engine (The Brain):** Runs pure Python logic with Zobrist hashing and Alpha-Beta pruning.
  * **GUI (The Hands):** A responsive Pygame interface with drag-and-drop and premove support.
  * **Interface (The Voice):** Supports CLI and UCI protocols for testing.

-----

## ⚡ Key Features (v1.0)

### 🧠 Intelligent Search

The core decision-making process uses advanced algorithms to calculate the best move efficiently.

  * **Negamax with Alpha-Beta Pruning:** Efficiently prunes the search tree to ignore bad variations.
  * **Principal Variation Search (PVS):** Optimizes search by assuming the first move is best, checking others with a zero window.
  * **Late Move Reduction (LMR):** Search less promising moves at reduced depth to save time.
  * **Null Move Pruning (NMP):** Massively increases search depth by pruning branches where passing the turn is still winning.
  * **Quiescence Search:** Solves the "Horizon Effect" by searching violent moves (captures) beyond the target depth.
  * **Iterative Deepening:** Ensures the engine always has a "best move" ready, even if time runs out.

### 🛡️ Strategic Evaluation

  * **Tapered Evaluation:** Interpolates between Middlegame and Endgame scores (e.g., King hides in middlegame, attacks in endgame).
  * **Piece-Square Tables (PST):** Complete tables for all pieces. The engine knows to centralize Knights/Bishops and put Rooks on open files.
  * **Threat Detection:** Static analysis prevents tactical blunders (like hanging pieces or moving a Queen to a square attacked by a Pawn).
  * **Bitboard Optimization:** Uses fast bitwise operations for analyzing Pawn Structures (Passers/Isolations).

### 🖥️ Modern GUI

  * **Drag & Drop:** Smooth piece movement mechanics.
  * **Pondering Arrow:** Visualizes what the engine thinks *you* will play next.
  * **Visual Promotion:** Context menu for selecting Queen, Rook, Bishop, or Knight.
  * **Premove Support:** Allows players to input moves during the engine's turn.

-----

## 🏗️ Architecture

### Engine Module (`engine/core/`)

| File | Role | Description |
| :--- | :--- | :--- |
| **`search.py`** | The "Eyes" | Implements Negamax, NMP, and Move Ordering (MVV-LVA). |
| **`bitboard_evaluator.py`** | The "Brain" | Handles material counting, PST lookup, and bitwise structure assessment. |
| **`loopboard_evaluator.py`** | The "Brain" | Handles material counting, PST lookup, with looping approach. |
| **`transposition.py`** | The "Memory"| Uses Zobrist Hashing (Polyglot) for $O(1)$ state lookups. |
| **`config.py`** | Settings | Centralized configuration for search depth, hash size, and weights. |

-----

## Installation

### Prerequisites

  * Python 3.13 or higher
  * pip package manager

### Setup Guide

```bash
# 1. Clone the repository
git clone [https://github.com/meedoomostafa/BlitzMate-engine.git](https://github.com/meedoomostafa/BlitzMate-engine.git)
cd BlitzMate-engine

# 2. Create virtual environment 
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip

# Install requirements for each module separately
pip install -r engine/requirements.txt
pip install -r gui/requirements.txt
pip install -r interface/requirements.txt

# 5. Download Tablebases (Critical for Endgame but optional)
# This script downloads ~500MB of Syzygy endgame data automatically
python setup_syzygy.py
```

-----

## 🎮 Usage

### Running the GUI (Recommended)

This launches the graphical board with **Depth 5** search enabled by default.

```bash
# from the root folder (BlitzMate-engine)
python -m gui.main
```

### Running the CLI

For testing or debugging without graphics:

```bash
# from the root folder (BlitzMate-engine) 
# interface still not implemented
python -m interface.cli
```

-----

## 🧠 Current Performance

  * **Search Depth:** Comfortably runs at Depth 5-6 in standard time controls.
  * **Nodes Per Second:** Optimized via Zobrist Hashing (Integers) vs old FEN strings.
  * **Style:** Positional/Tactical. Prioritizes development and safety; avoids "weird" shuffling.

-----

## 📋 To-Do Roadmap (v2.0)

**GUI & Architecture**

  * [x] **Non-Blocking Architecture:** Migrate the engine to a background thread so the window stays responsive during calculations.

**Optimization & Speed**

  * [x] **Bitboard Evaluation:** Migrate `evaluator.py` to use bitwise operations for pawn structures (Passers/Isolations).
  * [ ] **Multiprocessing:** Implement Lazy SMP to utilize multiple CPU cores (bypassing Python GIL).

**Knowledge**

  - [x] **Opening Book:** Integrate `chess.polyglot` to play standard openings (Sicilian, Queen's Gambit, etc.) instantly.
  * [x] **Endgame Tablebases:** Integrate Syzygy tablebases for perfect endgame play.

**Search Refinements**

  * [x] **Killer Heuristic:** Prioritize moves that caused cutoffs in sibling nodes.
  * [x] **Late Move Reduction (LMR):** Search less promising moves at reduced depth.

-----

## 🤝 Contributing

Contributions are welcome\! Feel free to:

1.  Report bugs (especially evaluation blind spots).
2.  Submit pull requests for optimization.

## 📄 License

This project is open source. Please check the repository for license details.

**👨‍💻 Author**
Created by **meedoomostafa**