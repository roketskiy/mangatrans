import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    checks = []
    with zipfile.ZipFile(args.source) as source, zipfile.ZipFile(args.output) as output:
        originals = [source.read(name) for name in source.namelist() if not name.endswith("/")]
        names = output.namelist()
        checks.append({"name": "page count retained", "passed": len(names) == len(originals)})
        checks.append({"name": "names unique and ordered", "passed": len(set(names)) == len(names) and names == sorted(names)})
        for name in names:
            data = output.read(name)
            webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
            checks.append({"name": name, "passed": (webp and name.lower().endswith(".webp")) or data in originals})
    before = hashlib.sha256(args.output.read_bytes()).hexdigest()
    result = subprocess.run(
        [str(args.engine.resolve()), "-in", str(args.source.resolve()), "-out", str(args.output.resolve())],
        capture_output=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    after = hashlib.sha256(args.output.read_bytes()).hexdigest()
    checks.append({"name": "unconfirmed overwrite rejected without changing output", "passed": result.returncode != 0 and before == after and b"output exists" in result.stderr})
    report = {"passed": all(check["passed"] for check in checks), "sha256": after, "checks": checks}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
