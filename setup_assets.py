"""Download opening book and Syzygy tablebase files for BlitzMate engine.

Usage
-----
    python setup_assets.py              # interactive – choose what to download
    python setup_assets.py books        # download ALL opening books
    python setup_assets.py syzygy       # download Syzygy tablebases
    python setup_assets.py all          # download everything (non-interactive)
"""

import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Opening books
# ---------------------------------------------------------------------------

BOOKS = [
    ("Titans.bin", "Titans – aggressive/tactical style (~1.8 MB)"),
    ("gm2600.bin", "GM2600 – grandmaster-level games (~340 KB)"),
    ("komodo.bin", "Komodo – Komodo engine openings (~8.8 MB)"),
    ("rodent.bin", "Rodent – Rodent engine openings (~2.7 MB)"),
    ("Human.bin", "Human – large human-games book (~14 MB)"),
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


def setup_books(selected: list[str] | None = None) -> None:
    """Download opening books from the GitHub release.

    Parameters
    ----------
    selected : list[str] | None
        Filenames to download.  ``None`` means *all* books.
    """
    os.makedirs(BOOKS_DIR, exist_ok=True)
    targets = selected if selected is not None else [b[0] for b in BOOKS]
    print(f"Downloading {len(targets)} opening book(s)...")
    for name in targets:
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


def _select_books_interactive() -> list[str] | None:
    """Show an interactive menu and return chosen book filenames.

    Returns ``None`` when the user picks "Download all".
    Returns an empty list when the user chooses to skip.
    """
    print("\n===== Opening Books =====")
    print("  0) Download ALL books")
    for idx, (name, desc) in enumerate(BOOKS, start=1):
        already = (
            " [installed]" if os.path.exists(os.path.join(BOOKS_DIR, name)) else ""
        )
        print(f"  {idx}) {desc}{already}")
    print(f"  {len(BOOKS) + 1}) Skip books\n")

    choice = input(
        "Enter your choices (comma-separated, e.g. 1,3) or press Enter for all: "
    ).strip()

    if not choice:
        return None  # default = all

    nums: list[int] = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            nums.append(int(part))
        else:
            print(f"  [!] Ignoring invalid input: {part}")

    if 0 in nums:
        return None  # all
    if (len(BOOKS) + 1) in nums:
        return []  # skip

    selected: list[str] = []
    for n in nums:
        if 1 <= n <= len(BOOKS):
            selected.append(BOOKS[n - 1][0])
        else:
            print(f"  [!] Ignoring out-of-range number: {n}")

    return selected if selected else None


def _interactive_menu() -> None:
    """Run the full interactive setup wizard."""
    print("=" * 50)
    print("  BlitzMate Asset Setup")
    print("=" * 50)
    print("\nWhat would you like to download?\n")
    print("  1) Opening books only")
    print("  2) Syzygy tablebases only")
    print("  3) Everything (books + Syzygy)")
    print("  4) Quit\n")

    choice = input("Choice [3]: ").strip() or "3"

    if choice == "4":
        print("Nothing to do. Bye!")
        return

    if choice in ("1", "3"):
        book_selection = _select_books_interactive()
        if book_selection is not None and len(book_selection) == 0:
            print("  Skipping books.\n")
        else:
            setup_books(book_selection)

    if choice in ("2", "3"):
        setup_syzygy()

    if choice not in ("1", "2", "3", "4"):
        print(f"Unknown choice: {choice}")
        sys.exit(1)

    print("All done!")


def main() -> None:
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == "books":
            setup_books()
        elif target == "syzygy":
            setup_syzygy()
        elif target == "all":
            setup_books()
            setup_syzygy()
        else:
            print(f"Unknown target: {target}")
            print("Usage: python setup_assets.py [books|syzygy|all]")
            sys.exit(1)
    else:
        _interactive_menu()


if __name__ == "__main__":
    main()
