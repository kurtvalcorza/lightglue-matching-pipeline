"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there. The router now carries the canonical refusal channel
(dat20-router-refusal/1), so the isolated runtime shows the same line. A genuine failure keeps its traceback."""
# ruff: noqa: E501

from __future__ import annotations

import json

import pytest

import lightglue_pipeline as lg
from test_lightglue_matching_colab_review_fixes import NOTEBOOK, _code_after, _section4, _zip

REFUSAL_CELL = "lightglue_matching-25"
HEADING = "## 4. iNaturalist photographs, homography pairs and split"


def _line(exc: BaseException) -> str:
    assert type(exc).__name__ == "BYODRefused" and exc.dimer_refusal is True and isinstance(exc, ValueError)
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    assert exc.__suppress_context__
    return rendered[0]


def test_the_scripted_refusal_input_is_one_line_naming_the_file(monkeypatch, tmp_path):
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    with pytest.raises(lg.BYODRefused) as caught:
        _section4(monkeypatch, tmp_path, byod_path=str(hostname))
    line = _line(caught.value)
    assert line.startswith("Refused: hostname: ") and "Run after" in line


def test_a_directory_and_a_missing_path_are_refused_by_name(monkeypatch, tmp_path):
    folder = tmp_path / "etc"
    folder.mkdir()
    (folder / "hostname").write_text("runtime-host\n", encoding="utf-8")
    for path, name in ((folder, "etc"), (tmp_path / "missing.zip", "missing.zip")):
        with pytest.raises(lg.BYODRefused) as caught:
            _section4(monkeypatch, tmp_path, byod_path=str(path))
        assert _line(caught.value).startswith(f"Refused: {name}: ")


def test_a_refusal_drops_the_section_4_results_of_an_earlier_input(monkeypatch, tmp_path):
    ok = _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "ok", 30)))
    assert ok["train_records"] and ok["splits"]
    source = _code_after(HEADING).replace("USE_BYOD = False", "USE_BYOD = True", 1)
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {str(tmp_path / 'missing.zip')!r}", 1)
    with pytest.raises(lg.BYODRefused) as caught:
        exec(compile(source, "<section 4>", "exec"), ok)
    assert _line(caught.value).startswith("Refused: missing.zip: ")
    assert not {"splits", "train_records", "val_records", "test_records", "data_source", "dataset_manifests"} & set(ok)


def test_a_genuine_failure_keeps_its_type(monkeypatch, tmp_path):
    def broken(path):
        raise RuntimeError("internal bug")

    monkeypatch.setattr(lg, "load_byod_dataset", broken)
    with pytest.raises(RuntimeError, match="internal bug"):
        _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "ok", 30)))


def test_the_refusal_cell_id_and_the_router_channel():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert source == _code_after(HEADING) and "BYOD_PATH = ''" in source and "refuse_byod(exc, file=byod_label)" in source
    router = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and "class IsolatedCellRefusal(Exception):" in "".join(c["source"]))
    assert 'if getattr(exc, "dimer_refusal", False):' in router and 'elif kind == "refusal":' in router
    assert 'return [f"Refused: {self}"]' in router
