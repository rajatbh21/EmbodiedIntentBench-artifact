"""Audit the public source tree before distributing a release.

Run from any directory; the default root is this script's checkout. Local caches,
runtime output and environments are excluded. Supply a private JSON array with
``--forbidden-file`` to search for private identifiers;
keep that array outside the release. Matching is case-insensitive and substring
based. Use ``--tracked-only`` on the final checkout to inspect exactly the files
that a Git source archive includes, including any tracked output or cache files.

Upstream credits must be preserved. If an identifier also belongs to an upstream
author, ``--attribution-allowlist`` accepts a JSON object mapping an exact relative
file path to an explanation. These explicit, reported exemptions affect only the
private identifier checks, never machine paths, size or symlink checks. The audit
does not inspect image pixels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit


EXCLUDED = frozenset({
    ".git", ".venv", "venv", "results", "output", ".cache", "cache",
    "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist",
})
# Conservative artifact packaging cap, not a claim about a hosting service limit.
MAX_BYTES = 8 * 1024 * 1024
# Build the expression without embedding a machine-specific example in this file.
MACHINE_PATH = re.compile(r"/(?:" + "|".join(("home", "Users")) + r")/[^\s\"'<>]+")
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|(?:\\.|[^\s)])+)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>]+>|\S+)", re.M)
FENCED_CODE = re.compile(r"^\s*(`{3,}|~{3,}).*$", re.M)


def markdown_targets(text: str) -> list[str]:
    """Extract ordinary inline/reference destinations outside fenced examples."""
    kept = []
    fence = None
    for line in text.splitlines():
        marker = FENCED_CODE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            kept.append(line)
    prose = "\n".join(kept)
    return [match.group(1).strip("<>")
            for pattern in (INLINE_LINK, REFERENCE_LINK)
            for match in pattern.finditer(prose)]


def audit_release(
    root: Path,
    *,
    forbidden: tuple[str, ...] = (),
    attribution_allowlist: dict[str, str] | None = None,
    max_bytes: int = MAX_BYTES,
    require_orphan: bool = False,
    tracked_only: bool = False,
) -> dict:
    """Return a deterministic audit result without writing to the checkout."""
    root = root.resolve()
    exemptions = attribution_allowlist or {}
    issues = []
    scanned = set()
    local_links = []
    byte_count = 0

    def fail(path: str, code: str, detail: str) -> None:
        issues.append({"path": path, "code": code, "detail": detail})

    if not root.is_dir():
        fail(".", "missing_root", "Release root is not a directory.")
    for path, reason in exemptions.items():
        p = Path(path)
        if p.is_absolute() or ".." in p.parts or not reason.strip():
            fail(path, "invalid_exemption", "Use an exact relative path and a nonempty reason.")

    candidates = []
    if tracked_only:
        try:
            checkout = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            if Path(checkout).resolve() != root:
                raise ValueError("Not the root of its own Git checkout.")
            names = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "-z", "--cached"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").split("\0")
            candidates = [root / name for name in sorted(set(names)) if name]
            if subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "--"],
                stderr=subprocess.DEVNULL,
            ).returncode != 0:
                fail(".git", "tracked_worktree_mismatch", "Tracked files differ from the Git index; stage the final public contents before auditing.")
        except (OSError, ValueError, subprocess.CalledProcessError):
            fail(".git", "missing_tracked_checkout", "Tracked inventory requires a standalone Git checkout.")
    else:
        for directory, subdirs, filenames in os.walk(root, followlinks=False):
            directory = Path(directory)
            subdirs[:] = sorted(d for d in subdirs if d not in EXCLUDED or (d in {"results", "output"} and directory != root))
            entries = sorted(filenames) + [d for d in subdirs if (directory / d).is_symlink()]
            candidates.extend(directory / name for name in entries if name not in EXCLUDED)

    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError):
                fail(relative, "broken_symlink", "Symlink is missing its target or is cyclic.")
                continue
            if not resolved.is_relative_to(root):
                fail(relative, "escaping_symlink", "Symlink target is outside the release.")
                continue
            if path.is_dir():
                continue
        if not path.is_file():
            fail(relative, "special_file", "Only ordinary files and internal symlinks are allowed.")
            continue
        scanned.add(relative)
        try:
            size = path.stat().st_size
            byte_count += size
            if size > max_bytes:
                fail(relative, "oversized_file", f"File is {size} bytes; limit is {max_bytes}.")
                continue
            data = path.read_bytes()
        except OSError:
            fail(relative, "unreadable_file", "Cannot read public file.")
            continue
        # Byte matching also detects identifiers in simple binary metadata.
        haystack = relative.casefold().encode() + b"\n" + data.lower()
        if relative not in exemptions:
            for index, token in enumerate(forbidden, 1):
                if token.casefold().encode() in haystack:
                    fail(relative, "forbidden_identifier", f"Matches private token {index}.")
        text = data.decode("utf-8", errors="replace")
        if MACHINE_PATH.search(text) or MACHINE_PATH.search(relative):
            fail(relative, "machine_path", "Contains an absolute user-home path.")
        if path.suffix.lower() == ".md":
            for target in markdown_targets(text):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                link = unquote(parsed.path)
                if link.startswith("/"):
                    destination = root / link.lstrip("/")
                else:
                    destination = path.parent / re.sub(r"\\(.)", r"\1", link)
                local_links.append((relative, target, destination))

    # Existing untracked files can conceal missing archive inputs. Resolve local
    # links only against the same public inventory that was actually inspected.
    for relative, target, destination in local_links:
        try:
            resolved = destination.resolve(strict=True)
        except (OSError, RuntimeError):
            fail(relative, "broken_link", f"Local destination does not exist: {target}")
            continue
        if not resolved.is_relative_to(root):
            fail(relative, "escaping_link", "Local destination is outside the release.")
            continue
        name = resolved.relative_to(root).as_posix()
        included = name in scanned or (resolved.is_dir() and any(
            item.startswith(name + "/") for item in scanned
        )) or (resolved == root and bool(scanned))
        if not included:
            fail(relative, "unpackaged_link", f"Local destination is outside the public file inventory: {target}")

    for path in exemptions:
        if path not in scanned:
            fail(path, "unused_exemption", "Exemption does not identify a scanned public file.")

    try:
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        project = metadata.get("project", {})
        if project.get("name") != "missionbench":
            fail("pyproject.toml", "package_name", "Project name must be missionbench.")
        entry = project.get("scripts", {}).get("missionbench", "")
        if not entry.startswith("missionbench.") or ":" not in entry:
            fail("pyproject.toml", "package_entrypoint", "The missionbench command must use its package.")
        authors = project.get("authors", [])
        if not authors or any(not isinstance(author, dict) or not author.get("name", "").strip()
                              for author in authors):
            fail("pyproject.toml", "author_metadata", "Package author metadata must name an author or contributor.")
        if not (root / "missionbench" / "__init__.py").is_file():
            fail("missionbench/__init__.py", "missing_package", "Package directory is missing.")
    except (OSError, ValueError):
        fail("pyproject.toml", "invalid_metadata", "Cannot read valid project metadata.")

    if require_orphan:
        try:
            checkout = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            if Path(checkout).resolve() != root:
                raise ValueError("Not the root of its own Git checkout.")
            commit = subprocess.check_output(
                ["git", "-C", str(root), "cat-file", "-p", "HEAD"],
                stderr=subprocess.DEVNULL, text=True,
            ).split("\n\n", 1)[0]
            if any(line.startswith("parent ") for line in commit.splitlines()):
                fail(".git", "history_parent", "Release commit has parent history.")
            if tracked_only and subprocess.run(
                ["git", "-C", str(root), "diff", "--cached", "--quiet", "HEAD", "--"],
                stderr=subprocess.DEVNULL,
            ).returncode != 0:
                fail(".git", "uncommitted_index", "The public index differs from the release commit used by Git archives.")
            identity = subprocess.check_output(
                ["git", "-C", str(root), "show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%B", "HEAD"],
                stderr=subprocess.DEVNULL, text=True,
            ).split("\0", 4)
            for index, token in enumerate(forbidden, 1):
                if token.casefold() in "\n".join(identity).casefold():
                    fail(".git", "forbidden_commit_identifier", f"Commit metadata matches private token {index}.")
        except (OSError, ValueError, subprocess.CalledProcessError):
            fail(".git", "missing_release_commit", "A committed standalone release checkout is required.")

    return {
        "passed": not issues,
        "scanned_files": len(scanned),
        "scanned_bytes": byte_count,
        "private_identifier_checks": len(forbidden),
        "orphan_checked": require_orphan,
        "inventory": "git_index" if tracked_only else "public_worktree",
        "attribution_exemptions": exemptions,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--forbidden-file", type=Path, help="Private JSON array of strings; keep outside the release.")
    parser.add_argument("--attribution-allowlist", type=Path, help="JSON object of exact upstream-credit paths and reasons.")
    parser.add_argument("--require-orphan", action="store_true")
    parser.add_argument("--tracked-only", action="store_true", help="Audit the Git index file inventory used for a source release.")
    parser.add_argument("--max-issues", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        if args.forbidden_file and args.forbidden_file.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("Keep the private forbidden-identifier file outside the release root.")
        forbidden = json.loads(args.forbidden_file.read_text()) if args.forbidden_file else []
        if not isinstance(forbidden, list) or any(not isinstance(x, str) or not x for x in forbidden):
            raise ValueError("Forbidden identifiers must be a JSON array of nonempty strings.")
        allowed = json.loads(args.attribution_allowlist.read_text()) if args.attribution_allowlist else {}
        if not isinstance(allowed, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in allowed.items()):
            raise ValueError("Attribution exemptions must map path strings to reason strings.")
        if args.max_issues < 0:
            raise ValueError("Maximum reported issues must be nonnegative.")
        result = audit_release(args.root, forbidden=tuple(forbidden), attribution_allowlist=allowed,
                               require_orphan=args.require_orphan, tracked_only=args.tracked_only)
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "configuration_error": str(error)}))
        return 2
    result["issue_count"] = len(result["issues"])
    result["issues"] = result["issues"][:args.max_issues]
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
