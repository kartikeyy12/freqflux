"""The browser page is one big script with no build step, so catch syntax slips here."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"


def _script() -> str:
    m = re.search(r"<script>(.*)</script>", PAGE.read_text(), re.S)
    assert m, "no inline script found in web/index.html"
    return m.group(1)


def test_page_script_has_no_syntax_errors(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    f = tmp_path / "page.js"
    f.write_text("(async()=>{" + _script() + "})")
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_page_has_the_controls_the_script_expects():
    html = PAGE.read_text()
    ids = set(re.findall(r"\$\('#([A-Za-z0-9_]+)'\)", _script()))
    declared = set(re.findall(r'id="([A-Za-z0-9_]+)"', html))
    # every #id the script looks up must exist in the markup (ids built dynamically are exempt)
    missing = sorted(i for i in ids if i not in declared and not i.startswith(("cc", "v_")))
    assert not missing, f"script references missing elements: {missing}"
