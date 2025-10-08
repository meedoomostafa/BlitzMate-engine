# BlitzMate Chess Engine ♟️

A high-performance chess engine built in Python with a modular architecture, featuring alpha-beta pruning, transposition tables, and an intuitive GUI.

## 🎯 Project Overview

BlitzMate is a classical chess engine implementation that combines efficient search algorithms with position evaluation to play competitive chess. The project is structured into three independent modules:

- **Engine**: Core chess logic, move generation, position evaluation, and search algorithms
- **GUI**: Graphical user interface for playing against the engine
- **Interface**: Communication layer supporting CLI, API, and UCI protocol

## 🏗️ Architecture

### Engine Module
- **`board.py`**: Chess board representation, move generation, and game state management
- **`evaluator.py`**: Position evaluation function considering material, piece placement, king safety, and more
- **`search.py`**: Alpha-beta pruning search with move ordering and iterative deepening
- **`transposition.py`**: Hash table for caching previously evaluated positions
- **`analyzer.py`**: Game analysis and position assessment tools
- **`config.py`**: Engine configuration and tunable parameters

### GUI Module
- Built with Python GUI framework
- Visual chess board with drag-and-drop piece movement
- Real-time position evaluation display
- Assets folder contains piece images (white/black pieces: Pawn, Knight, Bishop, Rook, Queen, King)

### Interface Module
- **`cli.py`**: Command-line interface for terminal play
- **`api.py`**: RESTful API for integration with other applications
- **`uci.py`**: Universal Chess Interface protocol support for compatibility with chess GUIs

## 🚀 Installation

### Prerequisites
- Python 3.13 or higher
- PyPy3 (recommended for engine performance)
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/meedoomostafa/BlitzMate-engine.git
cd BlitzMate-engine
```

### Setup Engine
```bash
cd engine
pypy3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

### Setup GUI
```bash
cd ../gui
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

### Setup Interface
```bash
cd ../interface
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

## 🎮 Usage

### Running the GUI
```bash
cd gui
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

### Running the CLI
```bash
cd interface
source venv/bin/activate  # On Windows: venv\Scripts\activate
python cli.py
```

### Using UCI Protocol
```bash
cd interface
source venv/bin/activate  # On Windows: venv\Scripts\activate
python uci.py
```

## 🧠 How It Works

### Move Generation
The engine uses bitboards and efficient data structures to generate all legal moves for a given position, including special moves like castling, en passant, and pawn promotion.

### Position Evaluation
The evaluator assigns a numerical score to each position based on:
- **Material balance**: Piece values (Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900)
- **Piece-square tables**: Positional bonuses for pieces on optimal squares
- **King safety**: Pawn shield and king exposure evaluation
- **Pawn structure**: Doubled, isolated, and passed pawns
- **Mobility**: Number of legal moves available
- **Control of key squares**: Center control and outpost evaluation

### Search Algorithm
The engine uses **alpha-beta pruning** with several optimizations:
- **Iterative deepening**: Gradually increases search depth for better time management
- **Move ordering**: Searches promising moves first (captures, checks, killers)
- **Transposition table**: Stores previously evaluated positions to avoid redundant calculations
- **Quiescence search**: Extends search for tactical positions to avoid horizon effect
- **Null move pruning**: Reduces search in positions where even passing doesn't help

## 📋 To-Do List

### High Priority
- [ ] **Optimize `evaluator.py`**: Improve evaluation function performance and accuracy
  - Profile bottlenecks in evaluation
  - Implement incremental evaluation updates
  - Fine-tune piece-square tables and weights
  
- [ ] **Optimize move caching**: Enhance transposition table efficiency
  - Implement better replacement schemes
  - Add aging mechanism for old entries
  - Optimize hash key generation

### Future Enhancements
- [ ] Opening book integration
- [ ] Endgame tablebases support
- [ ] Multi-threading for parallel search
- [ ] Machine learning-based evaluation
- [ ] Game notation (PGN) import/export
- [ ] Time control management
- [ ] Pondering (thinking on opponent's time)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source. Please check the repository for license details.

## 👨‍💻 Author

Created by [Meedo Mostafa](https://github.com/meedoomostafa)

---

**Note**: This engine is optimized to run with PyPy3 for maximum performance. The GUI and interface modules use standard Python 3.13 for better library compatibility.