"""
MFTMA manifold geometry analysis for the foveated-vision thesis.

Attempts to use the official neural_manifolds_replicaMFT package if available;
otherwise falls back to a sklearn-based approximation.

Public API
----------
manifold_radius          -- R_M: RMS distance from class centroid
manifold_dimension       -- D_M: participation ratio of within-class PCA eigs
manifold_capacity_approx -- alpha_c: random dichotomy LinearSVC fraction
compute_manifold_stats   -- all three metrics in one call
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

try:
    from mftma import manifold_analysis_corr as _mftma_official
    HAS_MFTMA = True
except ImportError:
    HAS_MFTMA = False


def manifold_radius(activations_by_class):
    """R_M: mean per-class RMS distance from centroid."""
    radii = []
    for acts in activations_by_class:
        mu = acts.mean(0, keepdims=True)
        radii.append(float(np.sqrt(((acts - mu)**2).sum(1).mean())))
    return float(np.mean(radii)) if radii else float('nan')


def manifold_dimension(activations_by_class):
    """D_M: mean participation ratio of within-class PCA eigenvalues."""
    dims = []
    for acts in activations_by_class:
        if acts.shape[0] < 2: continue
        c = acts - acts.mean(0)
        eigs = np.linalg.eigvalsh(c.T @ c / max(len(acts)-1, 1))
        eigs = eigs[eigs > 0]
        if len(eigs) == 0: continue
        dims.append(float((eigs.sum()**2) / (eigs**2).sum()))
    return float(np.mean(dims)) if dims else float('nan')


def manifold_capacity_approx(activations_by_class, n_dichotomies=20, seed=0):
    """alpha_c: fraction of random dichotomies solved by LinearSVC."""
    rng = np.random.RandomState(seed)
    n_cls = len(activations_by_class)
    feats = np.concatenate(activations_by_class, axis=0)
    ids   = np.concatenate([np.full(len(a), i) for i, a in enumerate(activations_by_class)])
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)
    split = int(0.8 * len(feats_s))
    ok = 0
    for _ in range(n_dichotomies):
        y = rng.choice([0, 1], size=n_cls)[ids]
        p = rng.permutation(len(feats_s))
        X_tr, y_tr = feats_s[p[:split]], y[p[:split]]
        X_te, y_te = feats_s[p[split:]], y[p[split:]]
        if len(set(y_tr)) < 2: continue
        try:
            clf = LinearSVC(max_iter=200, tol=1e-3)
            clf.fit(X_tr, y_tr)
            if (clf.predict(X_te) == y_te).mean() >= 0.95: ok += 1
        except Exception: pass
    return float(ok / n_dichotomies)


def compute_manifold_stats(activations_by_class, n_dichotomies=20, seed=0):
    """Compute all MFTMA metrics. activations_by_class: list of [M,d] arrays."""
    if HAS_MFTMA:
        # Use official neural_manifolds_replicaMFT
        kappa, alpha_c, R_M, D_M, *_ = _mftma_official(
            [a.T for a in activations_by_class], kappa=0.0, n_t=200)
        return {'alpha_c': float(alpha_c), 'R_M': float(R_M), 'D_M': float(D_M)}
    return {
        'alpha_c': manifold_capacity_approx(activations_by_class, n_dichotomies, seed),
        'R_M':     manifold_radius(activations_by_class),
        'D_M':     manifold_dimension(activations_by_class),
    }