from __future__ import annotations

from typing import Any, Dict

from pr2.editors.dummy import DummyEditor


def make_editor(editor_cfg: Dict[str, Any]):
    etype = editor_cfg.get("type", "dummy")
    if etype == "dummy":
        return DummyEditor(name=editor_cfg.get("name", "dummy"))
    if etype == "editor_a":
        raise NotImplementedError("editor_a adapter not implemented yet. Implement pr2/editors/editor_a.py")
    if etype == "editor_b":
        raise NotImplementedError("editor_b adapter not implemented yet. Implement pr2/editors/editor_b.py")
    raise ValueError(f"Unknown editor type: {etype}")
