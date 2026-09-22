"""The data registry. Offline: nothing here opens a socket."""
from __future__ import annotations

import os

import pytest

from lift import download


def test_every_entry_is_well_formed():
    for key, entry in download.REGISTRY.items():
        tier, url, source = entry
        assert not key.startswith("/"), f"{key} must be relative to the data directory"
        assert tier and url.startswith("https://"), key
        assert source, key


def test_urls_are_distinct():
    urls = [u for _, u, _ in download.REGISTRY.values()]
    assert len(urls) == len(set(urls))


def test_keys_resolve_under_the_project_data_directory():
    """projorg locates the project; this module hard-codes no path."""
    from projorg import gitdir

    root = os.path.join(gitdir(), "data")
    for key in download.REGISTRY:
        assert download.path_of(key).startswith(root)


def test_unknown_key_is_refused_before_any_network_use():
    with pytest.raises(RuntimeError, match="not a public artifact"):
        download.ensure("datasets/nothing/here.tar.gz")


def test_unknown_tier_is_refused():
    with pytest.raises(RuntimeError, match="unknown tier"):
        download.ensure_tier("nope")


def test_known_tiers_are_non_empty():
    for tier in {t for t, _, _ in download.REGISTRY.values()}:
        assert [k for k, (t, _, _) in download.REGISTRY.items() if t == tier]


def test_sources_table_lists_every_entry():
    table = download.sources()
    for key in download.REGISTRY:
        assert key in table
