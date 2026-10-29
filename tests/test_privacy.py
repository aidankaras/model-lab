"""Fails if personal data or private paths creep into the tree."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
SELF = Path(__file__).resolve()
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
OK_DOMAIN = re.compile(r"(^|\.)(example\.com|acme\.com)$")
# assembled so this file does not match itself
BANNED = ["/opt/" + "data", "/home/" + "vapten", "vacuum" + "one", "kar" + "as", "vap" + "ten", "i" + "cloud",
          "bu." + "edu", "tele" + "gram"]


def files():
    for p in ROOT.rglob("*"):
        if p.is_file() and p.name != "RELEASE-AUDIT.md" and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and p.resolve() != SELF:
            yield p


def text(p):
    try:
        return p.read_text()
    except UnicodeDecodeError:
        return ""


def test_no_real_email_addresses():
    bad = []
    for p in files():
        for m in EMAIL.findall(text(p)):
            if not OK_DOMAIN.search(m.split("@")[1].lower()):
                bad.append((str(p.relative_to(ROOT)), m))
    assert not bad, bad


def test_no_banned_strings():
    allowed = {"LICENSE", "README.md"}  # author name / badge URL only; checked separately
    bad = []
    for p in files():
        t = text(p).lower()
        for b in BANNED:
            if b in t and p.name not in allowed:
                bad.append((str(p.relative_to(ROOT)), b))
    assert not bad, bad


def test_readme_only_mentions_author_in_badge_urls():
    t = (ROOT / "README.md").read_text().lower()
    for line in t.splitlines():
        if "kar" + "as" in line:
            assert "github.com/aidan" in line


def test_no_private_data_files():
    names = {p.name for p in files()}
    assert not {n for n in names if n.startswith(("raw", "real-", "preds-", "vapt" + "en-", "pool-"))}
