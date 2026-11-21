"""Fails if personal data or private paths creep into the tree.

Banned terms are stored as truncated SHA-256 hashes of their lowercase form, so this file does not publish
the names it protects. To add a term: python -c "import hashlib;print(hashlib.sha256(b'term').hexdigest()[:16])"
"""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
SELF = Path(__file__).resolve()
# Vendored /git-signoff skill (jerrylin96/git-signoff): third-party files kept byte-identical; its
# author's contact details are his, not personal data from this project.
VENDORED = (".claude", "skills", "git-signoff")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
OK_DOMAIN = re.compile(r"(^|\.)(example\.com|acme\.com)$")

# substrings (private paths, hostnames, account names), keyed by length
BANNED_SUBSTRINGS = {
    5: {"43b9cbdbbd1017d8"},
    6: {"11c4ea6164fcb234", "e0aa81dd500c530a", "5b9466fa1fcc7d80"},
    8: {"3f40462915a3e602"},
    9: {"b7dd404ef0fcd08f", "50c97adc0c73aa1d"},
    12: {"22255525aa3b28a4"},
}
# whole words (internal codenames from the private setup the experiments ran in)
BANNED_WORDS = {"8cfde6efdfc4ed5a", "d6c1aaf96588e292", "d0960501f8971be8"}
AUTHOR_NAME = "43b9cbdbbd1017d8"  # allowed only in LICENSE, CITATION.cff and README badge URLs
BANNED_FILE_PREFIXES = ("raw", "real-", "preds-", "pool-")
BANNED_FILE_PREFIX_HASHES = {6: {"3c99b4419ead5e18"}}


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def banned_hits(text: str, substrings=BANNED_SUBSTRINGS, words=BANNED_WORDS) -> set[str]:
    """Hashes of banned terms found in text (case-insensitive)."""
    t = text.lower()
    found = {h(w) for w in re.findall(r"[a-z0-9]+", t)} & words
    for n, hashes in substrings.items():
        found |= {h(t[i:i + n]) for i in range(len(t) - n + 1)} & hashes
    return found


def files():
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).parts
        if rel[:3] == VENDORED:
            continue
        if p.is_file() and not (set(rel) & SKIP_DIRS) and p.resolve() != SELF:
            yield p


def text(p):
    try:
        return p.read_text()
    except UnicodeDecodeError:
        return ""


def test_banned_hits_mechanism():
    subs, words = {7: {h("abc/def")}}, {h("codename")}
    assert banned_hits("see /x/ABC/DEF/y", subs, words) == {h("abc/def")}
    assert banned_hits("the Codename here", subs, words) == {h("codename")}
    assert banned_hits("codenames and abcdef", subs, words) == set()


def test_no_real_email_addresses():
    bad = []
    for p in files():
        for m in EMAIL.findall(text(p)):
            if not OK_DOMAIN.search(m.split("@")[1].lower()):
                bad.append((str(p.relative_to(ROOT)), m))
    assert not bad, bad


def test_no_banned_strings():
    allowed = {"LICENSE", "README.md", "CITATION.cff"}  # author name / badge URLs only; checked separately
    bad = []
    for p in files():
        hits = banned_hits(text(p) + "\n" + str(p.relative_to(ROOT)))
        if p.name in allowed:
            hits -= {AUTHOR_NAME}
        if hits:
            bad.append((str(p.relative_to(ROOT)), sorted(hits)))
    assert not bad, bad


def test_readme_only_mentions_author_in_badge_urls():
    for line in (ROOT / "README.md").read_text().lower().splitlines():
        if AUTHOR_NAME in banned_hits(line, {5: {AUTHOR_NAME}}, set()):
            assert "github.com/aidan" in line, line


def test_no_private_data_files():
    names = {p.name for p in files()}
    assert not {n for n in names if n.startswith(BANNED_FILE_PREFIXES)}
    assert not {n for n in names if any(h(n[:k].lower()) in hs for k, hs in BANNED_FILE_PREFIX_HASHES.items())}
