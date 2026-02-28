# Contributing to BlitzMate

Thank you for your interest in contributing! This guide covers everything you need to get started.

> **Before writing any code**, read the [Architecture Guide](ARCHITECTURE.md) to understand how the modules connect.

---

## Getting Started

### Prerequisites

- Python 3.13+
- pip
- Git

### Setup

```bash
git clone https://github.com/meedoomostafa/BlitzMate-engine.git
cd BlitzMate-engine

python3.13 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: download opening books & tablebases
python setup_assets.py
```

### Verify

```bash
pytest tests/ -v
```

All tests must pass before submitting changes.

---

## Project Structure

```
engine/          Core logic — search, evaluation, config (no UI/IO)
interface/       Protocol adapters — UCI, REST API, CLI
gui/             PySide6 desktop application
tests/           pytest suites (unit + integration)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details on data flow, module boundaries, and algorithmic design.

---

## Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** — keep commits focused and atomic.

3. **Run tests** before pushing:
   ```bash
   pytest tests/ -v
   ```

4. **Push and open a Pull Request** against `main`.

---

## Code Conventions

### Python Style

- Follow **PEP 8**.
- Use **type hints** on all function signatures.
- Maximum line length: **100 characters** (soft limit, 120 hard).

### Comments & Docstrings

- **Module-level:** One-line docstring describing the file's purpose.
- **Class-level:** One-line docstring.
- **Method-level:** One-line docstring. Omit if the method name is self-explanatory.
- **Inline comments:** Use sparingly. Explain *why*, not *what*.
- No decorative comment dividers (`# ===`, `# ---`).
- End all comments with a period.

Example:
```python
"""Bitboard-based static evaluation with tapered scoring."""

class BitboardEvaluator:
    """Stateless position evaluator using 64-bit bitboard operations."""

    def evaluate(self, board: chess.Board) -> int:
        """Return static eval in centipawns, positive favors side to move."""
        ...
```

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Files | `snake_case` | `bitboard_evaluator.py` |
| Classes | `PascalCase` | `SearchEngine` |
| Functions / methods | `snake_case` | `search_best_move()` |
| Constants | `UPPER_SNAKE` | `MATE_SCORE` |
| Private methods | `_leading_underscore` | `_negamax()` |

### Imports

- Standard library → third-party → project modules (separated by blank lines).
- Prefer explicit imports over wildcards.

---

## Module Boundaries

These rules are **strict**:

| Module | Can import from |
|--------|----------------|
| `engine/` | Standard library, `chess` |
| `interface/` | `engine/` |
| `gui/` | `engine/` |

`engine/` must **never** import from `interface/` or `gui/`.

---

## Testing

### Where to add tests

| Test type | File | When |
|-----------|------|------|
| Unit (single component) | `tests/test_engine.py` | New evaluator feature, board method, TT behavior |
| Integration (multi-component) | `tests/test_integration.py` | Full game scenarios, UCI flow, API endpoints |

### Guidelines

- Every new eval feature needs at least one pair of positions proving it works (e.g., passed pawn scores higher than blocked pawn).
- Search changes need tactical position tests (mate-in-N, fork detection).
- Test names should describe the expected behavior: `test_passed_pawn_bonus`, `test_engine_finds_mate_in_2`.
- Use `pytest` fixtures and parameterize where appropriate.

### Running tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_engine.py -v

# Specific test
pytest tests/test_engine.py::TestBitboardEvaluator::test_passed_pawn_bonus -v
```

---

## Areas Open for Contribution

### Search Improvements
- Lazy SMP (multi-threaded search).
- Aspiration window tuning.
- Singular extensions.

### Evaluation Tuning
- Texel tuning for PSTs and weights.
- Connected pawn bonus.
- Rook on 7th rank bonus.
- Space advantage heuristic.

### Infrastructure
- `setoption` handling for Hash resize and Threads.
- Benchmark command (`go bench`).
- CI/CD pipeline with automated testing.
- SPRT testing framework for parameter tuning.

### Documentation
- Opening book format documentation.
- Endgame tablebase integration guide.

---

## Pull Request Checklist

- [ ] Branch is up to date with `main`.
- [ ] All existing tests pass (`pytest tests/ -v`).
- [ ] New functionality includes tests.
- [ ] Code follows the style conventions above.
- [ ] Comments are concise (see [Comments & Docstrings](#comments--docstrings)).
- [ ] No new dependencies added without discussion.
- [ ] Module boundary rules respected.

---

## Reporting Issues

When filing a bug:
1. Include the **FEN** of the position where the issue occurs.
2. Include the **depth** and any relevant config overrides.
3. Describe **expected vs actual** behavior.
4. If it's an evaluation issue, include the score breakdown if possible.

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).
