import os
import sys
from datetime import datetime
from typing import Tuple

# Local runner (no external deps)
from evaluation.octopack_runner import run_octopack_tests, _read_octopack_yaml


def main(repo_root: str = ".") -> int:
    cfg = _read_octopack_yaml(os.path.join(repo_root, "octopack.yaml"))
    if not cfg:
        print("[OctoPack] octopack.yaml not found or invalid in repo root.")
        return 2

    entry_point = os.path.join(repo_root, cfg["entry_point"])  # e.g., solution.py

    if not os.path.exists(entry_point):
        print(f"[OctoPack] Entry point file not found: {entry_point}\nPlace your solution code there and rerun.")
        return 3

    with open(entry_point, "r", encoding="utf-8") as f:
        code = f.read()

    passed, details = run_octopack_tests(code, root_dir=repo_root)

    # Print summary
    print("=== OctoPack pass@1 evaluation ===")
    print("Config:")
    print(f"  entry_point = {cfg['entry_point']}")
    print(f"  tests.path = {cfg.get('tests_path')}")
    print(f"  tests.input_pattern = {cfg.get('input_pattern')}")
    print(f"  tests.output_pattern = {cfg.get('output_pattern')}")
    print("Details:")
    for line in details:
        print("  ", line)
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    # Persist a brief report
    try:
        out_dir = os.path.join(repo_root, "run_result")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "octopack_pass1.txt"), "a", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"PASS@1: {'PASS' if passed else 'FAIL'}\n")
            f.write("\n".join(details) + "\n")
            f.write("-=*=-\n\n")
    except Exception as e:
        print(f"[WARN] Could not write report: {e}")

    return 0 if passed else 1


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(root))
