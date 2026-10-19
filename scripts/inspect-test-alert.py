import argparse
import json
import sqlite3
import shutil
import time
from pathlib import Path


def inspect(path: Path, deadline: float, snapshot: Path | None) -> dict:
    result = {"source": str(path), "matches": []}
    try:
        if snapshot:
            snapshot.mkdir(parents=True, exist_ok=True)
            copied = snapshot / path.name
            shutil.copyfile(path, copied)
            wal = path.with_name(path.name + "-wal")
            if wal.exists():
                shutil.copyfile(wal, copied.with_name(copied.name + "-wal"))
            path = copied
        with path.open("rb") as source:
            header = source.read(16)
        if header != b"SQLite format 3\x00":
            result["status"] = "not a SQLite database"
            return result
        uri = path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=2) as db:
            db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchmany(60)
            for (table,) in tables:
                quoted = '"' + table.replace('"', '""') + '"'
                columns = db.execute(f"PRAGMA table_info({quoted})").fetchall()
                names = [column[1] for column in columns]
                terms = [f"instr(lower(CAST(\"{name.replace(chr(34), chr(34) * 2)}\" AS TEXT)), ?) > 0" for name in names]
                if not terms:
                    continue
                rows = db.execute(f"SELECT * FROM {quoted} WHERE {' OR '.join(terms)} ORDER BY rowid DESC LIMIT 10", ["translate.test.exe"] * len(terms)).fetchall()
                for row in rows:
                    match = {"table": table, "fields": {}}
                    for name, value in zip(names, row):
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        if isinstance(value, str) and ("translate.test.exe" in value.lower() or "cobaltstrike" in value.lower()):
                            match["fields"][name] = value[:4096]
                    result["matches"].append(match)
        result["status"] = "read-only inspection complete"
    except (OSError, sqlite3.Error) as error:
        result["status"] = str(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    deadline = time.monotonic() + 10
    report = [inspect(path, deadline, args.snapshot) for path in args.sources]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
