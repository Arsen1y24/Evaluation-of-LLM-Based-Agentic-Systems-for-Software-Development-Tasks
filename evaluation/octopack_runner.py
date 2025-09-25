import os
import re
import subprocess
from glob import glob
from typing import Tuple, List, Dict, Optional


def _read_octopack_yaml(path: str = "octopack.yaml") -> Optional[Dict[str, str]]:
    """
    Very small, dependency-free parser for the specific octopack.yaml format used here.
    Expected keys:
      entry_point: solution.py
      tests:
        path: tests
        input_pattern: input*.txt
        output_pattern: output*.txt
        type: file
    Returns a flat dict with keys: entry_point, tests_path, input_pattern, output_pattern, type
    """
    if not os.path.exists(path):
        return None
    cfg: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            if line.startswith("entry_point:"):
                cfg["entry_point"] = line.split(":", 1)[1].strip()
            elif line.startswith("tests:"):
                i += 1
                # parse indented block
                while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t")):
                    tline = lines[i].strip()
                    if tline.startswith("path:"):
                        cfg["tests_path"] = tline.split(":", 1)[1].strip()
                    elif tline.startswith("input_pattern:"):
                        cfg["input_pattern"] = tline.split(":", 1)[1].strip()
                    elif tline.startswith("output_pattern:"):
                        cfg["output_pattern"] = tline.split(":", 1)[1].strip()
                    elif tline.startswith("type:"):
                        cfg["type"] = tline.split(":", 1)[1].strip()
                    i += 1
                continue  # skip the normal i += 1 for this branch
            i += 1
    except Exception:
        return None
    # sanity checks
    if not all(k in cfg for k in ("entry_point", "tests_path", "input_pattern", "output_pattern")):
        return None
    return cfg


def _match_expected(output_pattern: str, input_basename: str) -> Optional[str]:
    """
    Given output_pattern like 'output*.txt' and an input basename like 'input3.txt',
    return the expected output basename 'output3.txt'. Returns None if no number can be mapped.
    """
    m = re.search(r"(\d+)", input_basename)
    if not m:
        return None
    num = m.group(1)
    if "*" in output_pattern:
        return output_pattern.replace("*", num)
    # fallback: just prefix the number before extension
    root, ext = os.path.splitext(output_pattern)
    return f"{root}{num}{ext}"


def _normalize_output(s: str) -> str:
    # Normalize newlines and strip trailing spaces per line
    lines = s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    # Preserve final newline semantics similar to many judges: join with \n and strip final trailing newlines
    out = "\n".join(lines).rstrip("\n")
    return out


def run_octopack_tests(code: str, root_dir: str = ".") -> Tuple[bool, List[str]]:
    """
    Executes OctoPack-style file-based tests.
    - Writes the provided code into the configured entry_point (relative to root_dir).
    - For each tests/input*.txt, runs `python entry_point` with the file content as stdin.
    - Compares stdout with tests/output*.txt.
    Returns (all_passed, details[]) where details contains per-test results.
    """
    cfg = _read_octopack_yaml(os.path.join(root_dir, "octopack.yaml"))
    if not cfg:
        return False, ["octopack.yaml not found or invalid"]

    entry_point = os.path.join(root_dir, cfg["entry_point"])  # e.g., solution.py
    tests_dir = os.path.join(root_dir, cfg["tests_path"])     # e.g., tests
    input_pattern = cfg["input_pattern"]                      # e.g., input*.txt
    output_pattern = cfg["output_pattern"]                    # e.g., output*.txt

    os.makedirs(os.path.dirname(entry_point) or ".", exist_ok=True)

    # Write code to entry point
    with open(entry_point, "w", encoding="utf-8") as f:
        f.write(code)

    # Collect inputs
    inputs = sorted(glob(os.path.join(tests_dir, input_pattern)))
    if not inputs:
        return False, [f"No input files found in {tests_dir} matching {input_pattern}"]

    all_passed = True
    details: List[str] = []

    for in_path in inputs:
        base = os.path.basename(in_path)
        expected_base = _match_expected(output_pattern, base)
        if not expected_base:
            details.append(f"[SKIP] Could not map expected for {base}")
            all_passed = False
            continue
        out_path = os.path.join(tests_dir, expected_base)
        if not os.path.exists(out_path):
            details.append(f"[SKIP] Expected file missing: {out_path}")
            all_passed = False
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            input_data = f.read()
        with open(out_path, "r", encoding="utf-8") as f:
            expected = f.read()

        try:
            proc = subprocess.run(
                ["python", entry_point],
                input=input_data.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as e:
            details.append(f"[ERROR] Running {entry_point} failed for {base}: {e}")
            all_passed = False
            continue

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")

        norm_out = _normalize_output(stdout)
        norm_exp = _normalize_output(expected)

        if norm_out == norm_exp and proc.returncode == 0:
            details.append(f"[OK] {base}")
        else:
            all_passed = False
            snippet_err = (" | stderr: " + stderr.strip()) if stderr.strip() else ""
            details.append(
                f"[FAIL] {base} | expected: {repr(norm_exp)} | got: {repr(norm_out)}{snippet_err}"
            )

    return all_passed, details
