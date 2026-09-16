"""Point the committed MLflow store at the current checkout.

MLflow records artefact locations as absolute paths, so an ``mlflow.db``
created on one machine points at directories that do not exist after a clone.
This script rewrites every stored path prefix to the current project root.
It is idempotent and is run automatically by ``make ui`` and ``make serve``.

Usage::

    python scripts/relocate_mlflow.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "mlflow.db"
MARKER = "/mlruns"


def stored_root(conn: sqlite3.Connection) -> str | None:
    """Return the project root recorded in the store, if any."""
    row = conn.execute(
        "SELECT artifact_location FROM experiments WHERE artifact_location LIKE ? LIMIT 1",
        (f"%{MARKER}%",),
    ).fetchone()
    if not row:
        return None
    location = row[0].removeprefix("file://")
    return location[: location.index(MARKER)]


def relocate(conn: sqlite3.Connection, old: str, new: str) -> int:
    """Replace ``old`` with ``new`` in every text column of every table."""
    changed = 0
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        for column in conn.execute(f'PRAGMA table_info("{table}")').fetchall():
            name, kind = column[1], (column[2] or "").upper()
            if "CHAR" not in kind and "TEXT" not in kind and "CLOB" not in kind:
                continue
            cursor = conn.execute(
                f'UPDATE "{table}" SET "{name}" = REPLACE("{name}", ?, ?) '
                f'WHERE "{name}" LIKE ?',
                (old, new, f"%{old}%"),
            )
            changed += cursor.rowcount
    return changed


def main() -> None:
    """Rewrite the store in place if it was created elsewhere."""
    if not DB.exists():
        print("no mlflow.db yet - run `make all` first")
        return
    with sqlite3.connect(DB) as conn:
        old = stored_root(conn)
        new = str(ROOT)
        if old is None or old == new:
            print(f"mlflow.db already points at {new}")
            return
        print(f"relocating {old} -> {new}: {relocate(conn, old, new)} values updated")


if __name__ == "__main__":
    main()
