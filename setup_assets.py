"""Download opening book and Syzygy tablebase files for BlitzMate engine."""

import os
import urllib.request
import sys

# ---------------------------------------------------------------------------
# Opening books
# ---------------------------------------------------------------------------

BOOKS = [
    "Titans.bin",
    "gm2600.bin",
    "komodo.bin",
    "rodent.bin",
    "Human.bin",
]

BOOKS_BASE_URL = (
    "https://github.com/meedoomostafa/BlitzMate-engine"
    "/releases/download/v0.1-assets/"
)

BOOKS_DIR = os.path.join("engine", "assets", "openings")

# ---------------------------------------------------------------------------
# Syzygy tablebases
# ---------------------------------------------------------------------------

SYZYGY_NAMES = [
    "KBBvK",
    "KBvKB",
    "KNPvKN",
    "KPvK",
    "KQPvKN",
    "KQvKB",
    "KRNvK",
    "KRvK",
    "KBNvK",
    "KBvKN",
    "KNvK",
    "KPvKP",
    "KQPvKQ",
    "KQvKN",
    "KRPvK",
    "KRvKB",
    "KBPvK",
    "KBvKP",
    "KNvKN",
    "KQBvK",
    "KQPvKR",
    "KQvKP",
    "KRPvKB",
    "KRvKN",
    "KBPvKB",
    "KNNvK",
    "KNvKP",
    "KQNvK",
    "KQQvK",
    "KQvKQ",
    "KRPvKN",
    "KRvKP",
    "KBPvKN",
    "KNPvK",
    "KPPvK",
    "KQPvK",
    "KQRvK",
    "KQvKR",
    "KRPvKR",
    "KRvKR",
    "KBvK",
    "KNPvKB",
    "KPPvKP",
    "KQPvKB",
    "KQvK",
    "KRBvK",
    "KRRvK",
]

SYZYGY_BASE_URL = "https://tablebase.sesse.net/syzygy/3-4-5/"
SYZYGY_DIR = os.path.join("engine", "assets", "syzygy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: str) -> None:
    """Download *url* to *dest*, skipping if already present."""
    if os.path.exists(dest):
        print(f"  [Skip] {os.path.basename(dest)} (already exists)")
        return

    print(f"  [Down] {os.path.basename(dest)}...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print(" OK")
    except Exception as e:
        print(f" Error: {e}")


# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------


def setup_books() -> None:
    """Download opening books from the GitHub release."""
    os.makedirs(BOOKS_DIR, exist_ok=True)
    print(f"Checking {len(BOOKS)} opening book(s)...")
    for name in BOOKS:
        _download(BOOKS_BASE_URL + name, os.path.join(BOOKS_DIR, name))
    print("Opening-book setup complete.\n")


def setup_syzygy() -> None:
    """Download Syzygy 3-4-5 piece WDL + DTZ tables."""
    wdl_dir = os.path.join(SYZYGY_DIR, "wdl")
    dtz_dir = os.path.join(SYZYGY_DIR, "dtz")
    os.makedirs(wdl_dir, exist_ok=True)
    os.makedirs(dtz_dir, exist_ok=True)

    total = len(SYZYGY_NAMES) * 2
    print(f"Checking {total} Syzygy files (WDL + DTZ)...")
    for base in SYZYGY_NAMES:
        _download(
            SYZYGY_BASE_URL + f"{base}.rtbw",
            os.path.join(wdl_dir, f"{base}.rtbw"),
        )
        _download(
            SYZYGY_BASE_URL + f"{base}.rtbz",
            os.path.join(dtz_dir, f"{base}.rtbz"),
        )
    print("Syzygy setup complete.\n")


def main() -> None:
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == "books":
            setup_books()
        elif target == "syzygy":
            setup_syzygy()
        else:
            print(f"Unknown target: {target}")
            print("Usage: python setup_assets.py [books|syzygy]")
            sys.exit(1)
    else:
        setup_books()
        setup_syzygy()


if __name__ == "__main__":
    main()
