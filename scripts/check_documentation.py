"""Check tracked Markdown links and explicitly allowlisted read-only CLI interfaces.

No shell blocks are executed. Link checks use only the standard library and Git's
tracked-file index; private/untracked roadmap notes and datasets are not scanned.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(text: str) -> str:
    """Remove fenced code while retaining line positions and ordinary inline links."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            lines.append("")
        elif match:
            fence = match[1]
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def anchors(text: str) -> set[str]:
    """Collect GitHub-style ATX heading slugs, duplicate suffixes and HTML anchors."""
    clean = prose(text)
    found = set(re.findall(r"(?:id|name)=[\"\']([^\"\']+)[\"\']", clean))
    for heading in re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", clean, re.M):
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        heading = re.sub(r"<[^>]+>", "", heading).lower()
        slug = "".join(
            c for c in heading if c in " -_" or unicodedata.category(c)[0] in "LN"
        ).replace(" ", "-")
        unique = slug
        index = 0
        while unique in found:
            index += 1
            unique = f"{slug}-{index}"
        found.add(unique)
    return found


def check_presentation(text: str) -> list[str]:
    """Detect unclosed fences and Mermaid blocks without an adjacent prose paragraph."""
    errors = []
    lines = text.splitlines()
    fence = None
    mermaid = False
    for index, line in enumerate(lines):
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        if not match:
            continue
        if fence is None:
            fence = match[1]
            mermaid = match[2].strip() == "mermaid"
        elif match[1][0] == fence[0] and len(match[1]) >= len(fence):
            if mermaid:
                after = next((v.strip() for v in lines[index + 1 :] if v.strip()), "")
                if not after or after.startswith(("#", "```", "~~~", "|", "- ")):
                    errors.append(
                        "Mermaid diagram needs an immediately following prose explanation"
                    )
            fence = None
    if fence:
        errors.append("Unclosed code fence")
    return errors


def destinations(text: str) -> list[str]:
    """Read inline/image and reference-style destinations outside fenced code."""
    clean = prose(text)
    targets = re.findall(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", clean)
    targets += re.findall(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", clean, re.M)
    return [s.strip("<>") for s in targets]


def check_links(root: Path, paths: list[Path]) -> list[str]:
    """Validate local files/directories and Markdown heading fragments offline."""
    errors = []
    cache = {}
    for relative in paths:
        file = root / relative
        text = file.read_text()
        errors.extend(f"{relative}: {error}" for error in check_presentation(text))
        for link in destinations(text):
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (file.parent / unquote(url.path)).resolve() if url.path else file
            if not target.is_relative_to(root.resolve()):
                errors.append(f"{relative}: target escapes repository: {link}")
            elif not target.exists():
                errors.append(f"{relative}: missing local target: {link}")
            elif url.fragment and target.suffix.lower() == ".md":
                if target not in cache:
                    cache[target] = anchors(target.read_text())
                if unquote(url.fragment) not in cache[target]:
                    errors.append(f"{relative}: missing heading: {link}")
    return errors


def check_cli(root: Path) -> list[str]:
    """Run only named scripts with --help; never interpret documentation shell text."""
    errors = []
    cases = json.loads((root / "docs/examples/cli_interfaces.json").read_text())
    for case in cases:
        script = (root / case["script"]).resolve()
        if not script.is_relative_to(root) or script.parent != root / "scripts":
            errors.append(f"Unsafe script locator: {case['script']}")
            continue
        command = case.get("command", [])
        if not isinstance(command, list) or any(
            not isinstance(c, str) or not re.fullmatch("[a-z][a-z-]*", c) for c in command
        ):
            errors.append(f"Unsafe help subcommand: {command}")
            continue
        result = subprocess.run(
            [sys.executable, str(script), *command, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            errors.append(f"{case['script']}: --help failed: {result.stderr.strip()}")
        for option in case["options"]:
            if option not in set(re.findall(r"--[a-z][a-z0-9-]*", result.stdout)):
                errors.append(f"{case['script']}: missing documented option {option}")
            if not any(option in (root / doc).read_text() for doc in case["docs"]):
                errors.append(f"{case['script']}: option {option} absent from selected guide")
    return errors


def main() -> int:
    """Check tracked Markdown, optionally adding explicit CLI help checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-cli", action="store_true", help="Verify allowlisted --help interfaces"
    )
    args = parser.parse_args()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = [Path(p) for p in tracked if p.endswith(".md")]
    errors = check_links(ROOT, paths)
    if args.check_cli:
        errors += check_cli(ROOT)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Documentation links verified: {len(paths)} tracked Markdown files")
    if args.check_cli:
        print("Allowlisted CLI --help interfaces verified; no shell examples executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
