"""Recheck the two all-row p30 caption diagnostics without fitting."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree


def _scalar_rank_order(distances):
    # SIMD quicksort orders tied distances differently; isolate the compatible
    # scalar ranking so the predictive process keeps its original CPU settings.
    environment = os.environ.copy()
    disabled = set(filter(None, environment.get("NPY_DISABLE_CPU_FEATURES", "").split(",")))
    disabled.update(("AVX2", "AVX512F", "AVX512_SKX"))
    environment["NPY_DISABLE_CPU_FEATURES"] = ",".join(sorted(disabled))
    worker = (
        "import json, sys\n"
        "import numpy as np\n"
        "distances = np.asarray(json.load(sys.stdin))\n"
        "json.dump(np.argsort(distances, axis=1).tolist(), sys.stdout)\n"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", worker],
        input=json.dumps(distances.tolist()), text=True, capture_output=True,
        env=environment, cwd=Path(__file__).resolve().parent,
        check=True, timeout=60,
    )
    return np.asarray(json.loads(result.stdout), dtype=int)


def caption_trustworthiness(original, embedding):
    """Evaluate k10 for the archived all-row p30 caption coordinates."""
    original, embedding = np.asarray(original), np.asarray(embedding)
    n, k = len(original), 10
    if original.ndim != 2 or embedding.shape != (n, 2) or n <= 2 * k:
        raise ValueError("Caption diagnostics require more than 20 aligned rows and two embedding columns")
    if not np.isfinite(original).all() or not np.isfinite(embedding).all():
        raise ValueError("Caption diagnostics require finite SHAP values and coordinates")
    distances = pairwise_distances(original)
    np.fill_diagonal(distances, np.inf)
    order = _scalar_rank_order(distances)
    indices = KDTree(embedding, leaf_size=30).query(embedding, k=k + 1, return_distance=False)
    keep = indices != np.arange(n)[:, None]
    keep[keep.all(axis=1), 0] = False
    nearest = indices[keep].reshape(n, k)
    ranks = np.empty((n, n), dtype=int)
    ranks[np.arange(n)[:, None], order] = np.arange(1, n + 1)
    penalty = np.maximum(ranks[np.arange(n)[:, None], nearest] - k, 0).sum()
    return float(1 - penalty * (2 / (n * k * (2 * n - 3 * k - 1))))
