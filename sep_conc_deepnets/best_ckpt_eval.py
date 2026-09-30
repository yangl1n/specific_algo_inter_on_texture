"""Result keys for main.py's end-of-training evaluation of the best-val checkpoint.

``main.py`` evaluates an in-memory snapshot of the best-validation weights on the test
and val splits and writes the results into the run's ``results_*.json``; ``nc_report.py``
(and the offline test-accuracy report) read these keys instead of rebuilding the model.

    best_acc1                  max validation accuracy over epochs
    test_acc1                  last-epoch weights on test
    best_ckpt_test_acc1        best-val weights on test (comparable to deep_scatter,
                               which reloads best.pth before its test evaluation)
    best_ckpt_val_acc1         best-val weights on val
"""

import json
import math

# Bump when the evaluation semantics change, so runs written by an older main.py are
# recomputed offline. Compared with `>=` as a string, so keep the YYYY-MM-DD prefix.
BEST_CKPT_EVAL_VERSION = "2026-08-28"

SPLITS = ("test", "val")

VERSION_KEY = "best_ckpt_eval_version"
SOURCE_KEY = "best_ckpt_source"
EPOCH_KEY = "best_ckpt_epoch"
LOGGED_BEST_KEY = "best_acc1"

_ACC_FIELDS = ("acc1", "acc5", "loss", "n_samples")
# nc1_old is needed by nc_report to derive `n_features` (nc1_trace / nc1_old).
_NC_FIELDS = ("nc1_rel", "nc1_old", "nc1_trace", "between_class_trace", "nc2", "nc3")
# Optional skewed / Papyan NC1 (not written by main.py); a run lacking them is recomputed.
_NC_SKEW_FIELDS = ("nc1_skew", "nc1_skew_proj", "nc1_skew_mahal", "skew_rank", "nc1_papyan")


def acc_keys(split):
    """Result-key names for the accuracy of the best-val weights on ``split``."""
    return {f: "best_ckpt_{}_{}".format(split, f) for f in _ACC_FIELDS}


def nc_keys(split):
    """Result-key names for the neural-collapse metrics of the best-val weights."""
    return {f: "best_ckpt_{}_{}".format(split, f) for f in _NC_FIELDS}


def all_keys():
    """Every key this contract may write -- for documentation and for tests."""
    keys = [VERSION_KEY, SOURCE_KEY, EPOCH_KEY]
    for s in SPLITS:
        keys += list(acc_keys(s).values()) + list(nc_keys(s).values())
    return keys


# ---------------------------------------------------------------------------
# Reading side (nc_report.py and the offline test-accuracy report)
# ---------------------------------------------------------------------------

def _finite(v):
    """A key counts as present only if it holds a finite number (None/NaN -> recompute)."""
    if v is None or isinstance(v, bool):
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def read_results(results_json):
    """The ``results`` block of a run's JSON, or ``{}`` if unreadable."""
    try:
        with open(results_json) as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    results = payload.get("results", payload)
    return results if isinstance(results, dict) else {}


def version_ok(results, minimum=BEST_CKPT_EVAL_VERSION):
    """True when the run was written by a main.py at least this recent."""
    got = results.get(VERSION_KEY)
    return isinstance(got, str) and got >= minimum


def inline_acc(results, split):
    """``{acc1, acc5, loss, n_samples}`` if the run recorded them, else ``None``."""
    if not version_ok(results):
        return None
    keys = acc_keys(split)
    # acc5 is exempt: a <5-class dataset has no top-5, and main.py writes it as NaN there.
    required = ("acc1",)
    if not all(_finite(results.get(keys[f])) for f in required):
        return None
    return {f: (float(results[keys[f]]) if _finite(results.get(keys[f])) else None)
            for f in _ACC_FIELDS}


def inline_nc(results, split):
    """``{nc1_rel, nc1_trace, between_class_trace, nc2, nc3}`` or ``None``."""
    if not version_ok(results):
        return None
    keys = nc_keys(split)
    if not all(_finite(results.get(keys[f])) for f in _NC_FIELDS):
        return None
    return {f: float(results[keys[f]]) for f in _NC_FIELDS}


def inline_nc_skew(results, split):
    """``{nc1_skew, nc1_skew_proj, nc1_skew_mahal, skew_rank, nc1_papyan}`` or ``None``."""
    if not version_ok(results):
        return None
    keys = {f: "best_ckpt_{}_{}".format(split, f) for f in _NC_SKEW_FIELDS}
    if not all(_finite(results.get(keys[f])) for f in _NC_SKEW_FIELDS):
        return None
    return {f: float(results[keys[f]]) for f in _NC_SKEW_FIELDS}


def source_label(results):
    """Provenance string main.py recorded for the weights it evaluated."""
    return results.get(SOURCE_KEY) or "inline (main.py best-ckpt eval)"
