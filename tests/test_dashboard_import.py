#!/usr/bin/env python
"""Simple smoke test that the dashboard module imports and defines main()."""

import importlib


def test_dashboard_import():
    mod = importlib.import_module("examples.dashboard.app")
    assert hasattr(mod, "main")

