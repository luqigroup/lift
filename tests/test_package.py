"""Everything the release ships imports, and the scripts parse."""
from __future__ import annotations

import ast
import importlib
import pathlib
import pkgutil

import pytest

import lift

ROOT = pathlib.Path(lift.__file__).resolve().parent.parent
MODULES = sorted(m.name for m in pkgutil.walk_packages(lift.__path__, "lift."))
SCRIPTS = sorted((ROOT / "scripts").rglob("*.py"))


def test_the_package_has_modules():
    assert len(MODULES) > 40


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(path):
    ast.parse(path.read_text(), filename=str(path))


def test_no_machine_specific_paths():
    """A release must not reach into a directory only its author has."""
    offenders = []
    for path in list((ROOT / "lift").rglob("*.py")) + SCRIPTS:
        text = path.read_text()
        for needle in ("/home/", "/Users/", "C:\\\\"):
            if needle in text:
                offenders.append(f"{path.relative_to(ROOT)}: {needle}")
    assert not offenders, offenders


def test_public_api_is_importable():
    from lift import Lift, tag_positive_  # noqa: F401
