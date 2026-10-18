"""The code snippets of the user guide run.

All ``python`` code blocks of a guide page are executed in order in one namespace. They run in a
temporary working directory that contains a copy of ``examples/data``, so snippets refer to the
example data as from the repository root. Blocks written as ``{code-block} python`` are shown but
not executed.
"""

import os
import re
import shutil
from pathlib import Path

import pytest

GUIDE = Path(__file__).parents[1] / "docs" / "guide"
REPOSITORY = Path(__file__).parents[1]
_BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def pages() -> list[Path]:
    return sorted(p for p in GUIDE.rglob("*.md") if _BLOCK.search(p.read_text()))


@pytest.mark.slow
@pytest.mark.parametrize("page", pages(), ids=lambda p: str(p.relative_to(GUIDE)))
def test_snippets_run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(REPOSITORY / "examples" / "data", tmp_path / "examples" / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict[str, object] = {"__name__": "__guide__"}
    for index, block in enumerate(_BLOCK.findall(page.read_text())):
        code = compile(block, f"{page.name}[{index}]", "exec")
        exec(code, namespace)
    assert os.getcwd() == str(tmp_path)
