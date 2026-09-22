"""Phase 0 smoke tests: the project is wired up and the config is coherent.

These are cheap checks that run in CI on every commit. They catch typos in
params.yaml long before a training run would.
"""

from lending_club.config import PROJECT_ROOT, load_params, path_of


def test_project_root_contains_params():
    assert (PROJECT_ROOT / "params.yaml").exists()


def test_required_sections_present():
    params = load_params()
    for section in ("seed", "paths", "data", "split", "features", "profit", "model"):
        assert section in params, f"missing section: {section}"


def test_paths_resolve_under_project_root():
    assert path_of("raw_csv").is_relative_to(PROJECT_ROOT)


def test_allowlist_has_no_forbidden_columns():
    """The leakage guard that matters most: no post-issuance column may be a feature."""
    features = load_params()["features"]
    forbidden = tuple(features["forbidden_prefixes"])
    offenders = [c for c in features["allowlist"] if c.startswith(forbidden)]
    assert not offenders, f"post-issuance columns in allowlist: {offenders}"


def test_splits_do_not_overlap():
    split = load_params()["split"]
    assert split["train"]["end"] < split["validation"]["start"]
    assert split["validation"]["end"] < split["test"]["start"]
