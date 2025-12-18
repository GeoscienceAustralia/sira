"""Utility to list all external packages imported anywhere in the SIRA codebase.

This scans every ``*.py`` file under the repository root using ``ast`` and collects
top-level imported package names. Relative imports and the local project package
``sira`` are ignored. The output helps reconcile declared dependencies with actual
usage.

Usage (from repo root):

    python -m sira.tools.find_imported_modules

Or directly:

    python sira/tools/find_imported_modules.py

The script prints a sorted list of unique package names.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


def list_all_installed_packages():
    from importlib.metadata import distributions

    for dist in distributions():
        print(f"{dist.metadata['Name']}=={dist.version}")


def iter_python_files(root: Path):
    """Yield all Python file paths under ``root`` excluding common ignore dirs."""
    ignore_dirs = {".git", "__pycache__", ".venv", "venv"}
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for fname in filenames:
            if fname.endswith(".py"):
                yield Path(dirpath) / fname


def extract_imports(py_file: Path):
    """Return a set of top-level imported package names from a Python file.

    Uses ``ast`` for robust parsing. For ``import a.b.c`` or ``from a.b import x`` we
    record ``a``. Relative imports (``from .foo import bar``) are skipped.
    """
    pkgs: set[str] = set()
    try:
        src = py_file.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return pkgs
    try:
        tree = ast.parse(src, filename=str(py_file))
    except SyntaxError:
        return pkgs
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                pkgs.add(top)
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports
            if node.level and node.level > 0:
                continue
            if node.module:
                top = node.module.split(".")[0]
                pkgs.add(top)
    return pkgs


def is_stdlib(module: str) -> bool:
    """Heuristic check if ``module`` is part of the standard library.

    We attempt an import; if successful and its ``__file__`` resides in the
    Python install's ``lib`` directory (and not site-packages), we consider it
    stdlib. Failures or missing ``__file__`` default to stdlib status to reduce
    false positives.
    """
    if module in sys.builtin_module_names:
        return True
    try:
        mod = __import__(module)
    except Exception:
        # If it can't be imported we treat it as non-stdlib (likely external or local)
        return False
    path = getattr(mod, "__file__", "") or ""
    if not path:
        return True
    path_lower = path.lower()
    return ("site-packages" not in path_lower) and ("dist-packages" not in path_lower)


def collect_imported_packages(root: Path, exclude_stdlib: bool = True) -> list[str]:
    """Collect unique imported top-level package names.

    Parameters
    ----------
    root : Path
        Repository root to scan.
    exclude_stdlib : bool
        If True, filter out detected standard library modules.
    """
    all_pkgs: set[str] = set()
    for py_file in iter_python_files(root):
        all_pkgs.update(extract_imports(py_file))
    # Remove project local name and dunder future
    all_pkgs.discard("sira")
    all_pkgs.discard("__future__")
    if exclude_stdlib:
        all_pkgs = {p for p in all_pkgs if not is_stdlib(p)}
    return sorted(all_pkgs)


def main():
    repo_root = Path(__file__).resolve().parents[2]  # .../sira (repo root)
    pkgs = collect_imported_packages(repo_root)
    for p in pkgs:
        print(p)


if __name__ == "__main__":  # pragma: no cover
    main()
