import importlib
from importlib.util import find_spec
import sys
from unittest.mock import patch

import pytest

from test_main import _base_argv, _main_patches, _make_model_files


def test_importing_pipeline_does_not_patch_frozen_torch():
    orig = sys.modules.pop("jasna.pipeline", None)
    try:
        with patch("jasna._frozen.patch_frozen_torch") as spy:
            importlib.import_module("jasna.pipeline")
        spy.assert_not_called()
    finally:
        if orig is not None:
            sys.modules["jasna.pipeline"] = orig
        else:
            sys.modules.pop("jasna.pipeline", None)


def test_cli_main_patches_frozen_torch(tmp_path):
    inp, out, rest, det = _make_model_files(tmp_path)
    with patch("jasna._frozen.patch_frozen_torch") as spy:
        with _main_patches():
            with patch.object(sys, "argv", _base_argv(inp, out, rest, det)):
                from jasna.main import main
                main()
    spy.assert_called()


@pytest.mark.skipif(find_spec("tkinter") is None, reason="python3-tk is not installed")
def test_gui_run_gui_patches_frozen_torch():
    import customtkinter as ctk

    from jasna.gui import app as gui_app

    class _Stop(Exception):
        pass

    try:
        with patch("jasna._frozen.patch_frozen_torch") as spy:
            with patch.object(gui_app, "JasnaApp", side_effect=_Stop):
                with pytest.raises(_Stop):
                    gui_app.run_gui()
    finally:
        # run_gui() configures process-global CTk scaling before constructing
        # the app. The deliberate constructor failure must not leak that state
        # into later GUI tests.
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
    spy.assert_called()
