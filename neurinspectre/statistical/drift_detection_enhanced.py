"""
NeurInSpectre Enhanced Drift Detection Module
Implements advanced statistical drift detection methods
Intended for drift detection on real analysis inputs.

Key Features:
- Hotelling's T² test for multivariate drift detection
- Kolmogorov-Smirnov tests for distribution drift
- Bayesian change point detection with posterior probabilities
- Temporal drift analysis with sliding windows
- Statistical significance testing with Z-score analysis
"""

import numpy as np
from scipy import stats
from scipy.linalg import inv, LinAlgError
from sklearn.covariance import MinCovDet
from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def _bonferroni_adjust(p_values: List[float], *, m: Optional[int] = None) -> List[float]:
    """
    Bonferroni p-value adjustment (FWER control).

    Given a family of m hypothesis tests with raw p-values p_i, the Bonferroni-adjusted
    p-values are:
        p_i_adj = min(p_i * m, 1)
    """
    finite = []
    for p in p_values:
        try:
            p_f = float(p)
        except Exception:
            p_f = 1.0
        if not np.isfinite(p_f):
            p_f = 1.0
        finite.append(float(np.clip(p_f, 0.0, 1.0)))
    m_eff = int(m) if m is not None else len(finite)
    m_eff = max(1, m_eff)
    return [float(min(1.0, p * m_eff)) for p in finite]


def _bh_fdr_adjust(p_values: List[float]) -> List[float]:
    """
    Benjamini–Hochberg (BH) adjustment (FDR control).

    Returns BH-adjusted q-values in the original input order.
    """
    finite: List[float] = []
    for p in p_values:
        try:
            p_f = float(p)
        except Exception:
            p_f = 1.0
        if not np.isfinite(p_f):
            p_f = 1.0
        finite.append(float(np.clip(p_f, 0.0, 1.0)))

    m = int(len(finite))
    if m <= 0:
        return []

    order = np.argsort(np.asarray(finite, dtype=np.float64), kind="mergesort")
    ranked = [finite[int(i)] for i in order]

    # Raw BH: q_i = p_i * m / rank_i
    q = [0.0] * m
    for r, p in enumerate(ranked, start=1):
        q[r - 1] = float(min(1.0, float(p) * float(m) / float(r)))

    # Enforce monotonicity from the tail: q_(i) = min_{j>=i} q_j
    for i in range(m - 2, -1, -1):
        q[i] = float(min(q[i], q[i + 1]))

    out = [1.0] * m
    for pos, idx in enumerate(order):
        out[int(idx)] = float(q[int(pos)])
    return out


def _fisher_combine_p_values(p_values: List[float]) -> Tuple[float, float]:
    """
    Fisher's method for combining p-values.

    Returns:
        (statistic, combined_p_value)
    """
    eps = 1e-300
    ps: List[float] = []
    for p in p_values:
        try:
            pv = float(p)
        except Exception:
            pv = 1.0
        if not np.isfinite(pv):
            pv = 1.0
        pv = float(np.clip(pv, eps, 1.0))
        ps.append(pv)
    k = int(len(ps))
    if k <= 0:
        return 0.0, 1.0
    stat = float(-2.0 * float(np.sum(np.log(np.asarray(ps, dtype=np.float64)))))
    p_combined = float(stats.chi2.sf(stat, df=2 * k))
    p_combined = float(np.clip(p_combined, 0.0, 1.0))
    return stat, p_combined


@dataclass
class DriftDetectionResults:
    """Container for drift detection results"""
    drift_detected: bool
    drift_score: float
    p_value: float
    confidence_interval: Tuple[float, float]
    change_points: List[int]
    drift_magnitude: float
    statistical_significance: Dict[str, Any]
    feature_drift_scores: np.ndarray

class BaseDriftDetector(ABC):
    """Base class for drift detectors"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """Detect drift between reference and current data"""
        pass

class HotellingT2DriftDetector(BaseDriftDetector):
    """
    Hotelling's T² test for multivariate drift detection
    Based on 2024 research: "Temporal cross-validation impacts multivariate time series"
    """
    
    def __init__(self, confidence_level: float = 0.95, use_robust_covariance: bool = True):
        super().__init__(confidence_level)
        self.use_robust_covariance = use_robust_covariance
        self._last_p_value_method: str = "f"
        # Diagnostics captured during covariance estimation (kept in results JSON).
        self._last_covariance_estimator: str = "pooled"
        self._last_covariance_warnings: List[str] = []
        self._last_covariance_error: Optional[str] = None
        
    def compute_hotelling_t2(
        self, 
        X1: np.ndarray, 
        X2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute Hotelling's T² statistic
        
        Args:
            X1: Reference data matrix (n1, p)
            X2: Current data matrix (n2, p)
            
        Returns:
            T² statistic, F-statistic, p-value
        """
        n1, p = X1.shape
        n2, _ = X2.shape

        # Need at least 2 samples per group to form a covariance estimate.
        if n1 < 2 or n2 < 2 or (n1 + n2 - 2) <= 0:
            self._last_p_value_method = "insufficient_samples"
            return 0.0, 0.0, 1.0
        
        # Compute sample means
        mean1 = np.mean(X1, axis=0)
        mean2 = np.mean(X2, axis=0)
        
        # Compute pooled covariance matrix
        self._last_covariance_estimator = "pooled"
        self._last_covariance_warnings = []
        self._last_covariance_error = None
        if self.use_robust_covariance and min(n1, n2) > p:
            try:
                # Use robust covariance estimation
                combined_data = np.vstack([X1, X2])
                cov_estimator = MinCovDet(random_state=42)
                # MinCovDet can emit warnings on degenerate/low-rank data. Capture them
                # (so tests/CLI stay clean) and fall back to the classical pooled
                # covariance when the fit is unstable.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    cov_estimator.fit(combined_data)
                self._last_covariance_warnings = [str(w.message) for w in caught]
                if any("not full rank" in str(w.message).lower() for w in caught):
                    raise ValueError("MinCovDet covariance not full rank; falling back to pooled covariance")

                S_pooled = cov_estimator.covariance_
                self._last_covariance_estimator = "mincovdet"
            except Exception as e:
                self._last_covariance_error = str(e)
                logger.warning("Robust covariance estimation failed: %s. Using empirical.", e)
                S1 = np.cov(X1, rowvar=False, bias=False)
                S2 = np.cov(X2, rowvar=False, bias=False)
                S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
                self._last_covariance_estimator = "pooled_fallback"
        else:
            # Standard pooled covariance
            S1 = np.cov(X1, rowvar=False, bias=False)
            S2 = np.cov(X2, rowvar=False, bias=False)
            S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
        
        # Compute T² statistic
        try:
            S_inv = inv(S_pooled)
            diff = mean1 - mean2
            T2 = (n1 * n2) / (n1 + n2) * np.dot(np.dot(diff, S_inv), diff)
        except LinAlgError:
            # Regularize covariance matrix if singular
            S_regularized = S_pooled + np.eye(p) * 1e-6
            S_inv = inv(S_regularized)
            diff = mean1 - mean2
            T2 = (n1 * n2) / (n1 + n2) * np.dot(np.dot(diff, S_inv), diff)
        
        # Convert to an F-statistic when degrees of freedom are valid; otherwise use a
        # chi-square approximation (T² ~ χ²_p) to avoid invalid/NaN p-values for small samples.
        df1 = p
        df2 = n1 + n2 - p - 1
        denom = float((n1 + n2 - 2) * p)
        if df2 > 0 and denom > 0.0:
            F_stat = float(T2) * float(df2) / denom
            p_value = float(1.0 - stats.f.cdf(F_stat, df1, df2))
            self._last_p_value_method = "f"
        else:
            F_stat = 0.0
            p_value = float(1.0 - stats.chi2.cdf(float(T2), df=p))
            self._last_p_value_method = "chi2_approx"

        p_value = float(np.clip(p_value, 0.0, 1.0))
        
        return T2, F_stat, p_value
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """
        Detect drift using Hotelling's T² test
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            DriftDetectionResults object
        """
        logger.info(f"Hotelling's T² drift detection: ref={reference_data.shape}, cur={current_data.shape}")
        
        # Compute Hotelling's T² test
        T2, F_stat, p_value = self.compute_hotelling_t2(reference_data, current_data)
        
        # Determine if drift is detected
        drift_detected = p_value < self.alpha
        
        # Acceptance region under the null (at the configured confidence level):
        # drift indicated when T² > T²_critical.
        n1, p = reference_data.shape
        n2, _ = current_data.shape
        df1 = p
        df2 = n1 + n2 - p - 1
        
        # Critical value under the same p-value method used above.
        if getattr(self, "_last_p_value_method", "f") == "f" and df2 > 0:
            f_critical = stats.f.ppf(1 - self.alpha, df1, df2)
            T2_critical = f_critical * (n1 + n2 - 2) * p / (n1 + n2 - p - 1)
        else:
            # Chi-square approximation for small-sample/invalid df2 regime.
            T2_critical = float(stats.chi2.ppf(1 - self.alpha, df=p))
        
        ci_lower = 0.0
        ci_upper = float(T2_critical)
        
        # Feature-wise drift scores: standardized mean differences (robust, deterministic).
        feature_drift_scores = []
        for i in range(reference_data.shape[1]):
            x1 = reference_data[:, i]
            x2 = current_data[:, i]
            m1 = float(np.mean(x1))
            m2 = float(np.mean(x2))
            # Pooled std (ddof=1). If too few samples, fall back to a safe denominator.
            s1 = float(np.std(x1, ddof=1)) if x1.size >= 2 else 0.0
            s2 = float(np.std(x2, ddof=1)) if x2.size >= 2 else 0.0
            denom = 0.0
            if x1.size >= 2 and x2.size >= 2 and (x1.size + x2.size - 2) > 0:
                sp2 = (((x1.size - 1) * (s1 ** 2)) + ((x2.size - 1) * (s2 ** 2))) / float(x1.size + x2.size - 2)
                denom = float(np.sqrt(max(0.0, sp2)))
            else:
                # With tiny samples, use the combined (population) std as a reasonable scale.
                combined = np.concatenate([x1, x2], axis=0)
                denom = float(np.std(combined, ddof=0))
            denom = denom if denom > 1e-12 else 1.0
            feature_drift_scores.append(abs(m1 - m2) / denom)
        feature_drift_scores = np.array(feature_drift_scores)
        
        # Statistical significance measures
        statistical_significance = {
            'hotelling_t2': T2,
            'f_statistic': F_stat,
            'p_value': p_value,
            'p_value_method': getattr(self, "_last_p_value_method", "f"),
            'covariance_estimator': getattr(self, "_last_covariance_estimator", "pooled"),
            'covariance_warnings': list(getattr(self, "_last_covariance_warnings", [])),
            'covariance_error': getattr(self, "_last_covariance_error", None),
            'degrees_freedom_1': df1,
            'degrees_freedom_2': df2,
            't2_critical': float(T2_critical),
            'sample_sizes': (int(n1), int(n2)),
            'n_features': int(p),
        }
        
        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=T2,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            change_points=[],  # Not applicable for two-sample test
            drift_magnitude=T2,
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores
        )

class KolmogorovSmirnovDriftDetector(BaseDriftDetector):
    """
    Kolmogorov-Smirnov test for distribution drift detection
    """
    
    def __init__(self, confidence_level: float = 0.95, multivariate_method: str = 'energy'):
        super().__init__(confidence_level)
        self.multivariate_method = multivariate_method
        
    def ks_test_multivariate(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        *,
        return_details: bool = False,
        n_permutations: int = 1000,
        random_state: Optional[int] = 42,
    ) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        """
        Multivariate two-sample test using energy statistics (energy distance).
        (Note: this is not a literal multivariate KS test; it is an energy-based
        two-sample test commonly used for distribution drift detection.)
        
        Args:
            X1: Reference data
            X2: Current data
            
        Returns:
            (statistic, p_value) or (statistic, p_value, critical_value) if return_details=True.
        """
        from scipy.spatial.distance import pdist, squareform
        
        n1, n2 = len(X1), len(X2)
        
        if self.multivariate_method == 'energy':
            # Energy statistics approach
            # Compute pairwise distances
            combined = np.vstack([X1, X2])
            distances = squareform(pdist(combined))
            
            # Energy statistic
            def energy_stat(data1_idx, data2_idx):
                d11 = distances[np.ix_(data1_idx, data1_idx)]
                d22 = distances[np.ix_(data2_idx, data2_idx)]
                d12 = distances[np.ix_(data1_idx, data2_idx)]

                def mean_offdiag(d: np.ndarray) -> float:
                    n = d.shape[0]
                    if n <= 1:
                        return 0.0
                    return float((np.sum(d) - np.trace(d)) / (n * (n - 1)))

                term1 = mean_offdiag(d11)
                term2 = mean_offdiag(d22)
                term3 = float(np.mean(d12))

                stat = 2.0 * term3 - term1 - term2
                return float(max(0.0, stat))
            
            idx1 = np.arange(n1)
            idx2 = np.arange(n1, n1 + n2)
            
            observed_stat = energy_stat(idx1, idx2)
            
            # Permutation test for p-value / critical value
            rng = np.random.default_rng(random_state)
            permutation_stats = []
            
            for _ in range(n_permutations):
                combined_idx = rng.permutation(n1 + n2)
                perm_idx1 = combined_idx[:n1]
                perm_idx2 = combined_idx[n1:]
                perm_stat = energy_stat(perm_idx1, perm_idx2)
                permutation_stats.append(perm_stat)

            perm = np.asarray(permutation_stats, dtype=np.float64)
            p_value = float((1.0 + np.sum(perm >= observed_stat)) / (n_permutations + 1.0))
            critical_value = float(np.quantile(perm, 1.0 - self.alpha))

            if return_details:
                return observed_stat, p_value, critical_value
            return observed_stat, p_value
        else:
            # Fallback to univariate KS test on principal component
            from sklearn.decomposition import PCA
            
            # Project to first principal component
            pca = PCA(n_components=1)
            X1_proj = pca.fit_transform(X1).flatten()
            X2_proj = pca.transform(X2).flatten()
            
            # Standard KS test
            ks_stat, p_value = stats.ks_2samp(X1_proj, X2_proj)
            # Approximate two-sample KS critical value (two-sided).
            c_alpha = np.sqrt(-0.5 * np.log(self.alpha / 2.0))
            critical_value = float(c_alpha * np.sqrt((n1 + n2) / (n1 * n2)))

            if return_details:
                return float(ks_stat), float(p_value), critical_value
            return float(ks_stat), float(p_value)
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            DriftDetectionResults object
        """
        logger.info(f"KS drift detection: ref={reference_data.shape}, cur={current_data.shape}")
        
        # Multivariate test + critical value for the configured method
        ks_stat, p_value, critical_value = self.ks_test_multivariate(
            reference_data,
            current_data,
            return_details=True,
        )
        
        # Determine if drift is detected
        drift_detected = p_value < self.alpha
        
        # Feature-wise KS tests
        feature_drift_scores = []
        for i in range(reference_data.shape[1]):
            feature_ks_stat, _ = stats.ks_2samp(reference_data[:, i], current_data[:, i])
            feature_drift_scores.append(feature_ks_stat)
        feature_drift_scores = np.array(feature_drift_scores)
        
        # Acceptance region under the null (at the configured confidence level).
        # Drift indicated when statistic > critical_value.
        n1, n2 = len(reference_data), len(current_data)
        ci_lower = 0.0
        ci_upper = float(critical_value)
        
        # Statistical significance measures
        statistical_significance = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'sample_sizes': (int(n1), int(n2)),
            'n_features': int(reference_data.shape[1]) if reference_data.ndim == 2 else 1,
        }
        
        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=ks_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            change_points=[],
            drift_magnitude=ks_stat,
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores
        )


class MMDDriftDetector(BaseDriftDetector):
    """
    Maximum Mean Discrepancy (MMD) two-sample test for multivariate drift.

    We compute an unbiased MMD^2 estimate with an RBF kernel and obtain a p-value
    via a permutation test. This is AE-friendly (no questionable independence
    assumptions across deep feature dimensions) and matches the common
    recommendation to use multivariate two-sample tests (MMD / energy distance).
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        *,
        n_permutations: int = 500,
        bandwidth: Optional[float] = None,
        random_state: int = 42,
    ):
        super().__init__(confidence_level)
        self.n_permutations = int(max(50, n_permutations))
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.random_state = int(random_state)

    @staticmethod
    def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.ndim != 2:
            A = A.reshape(A.shape[0], -1)
        if B.ndim != 2:
            B = B.reshape(B.shape[0], -1)
        aa = np.sum(A * A, axis=1, keepdims=True)
        bb = np.sum(B * B, axis=1, keepdims=True).T
        d2 = aa + bb - 2.0 * (A @ B.T)
        return np.maximum(0.0, d2)

    @staticmethod
    def _median_bandwidth(X: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            return 1.0
        d2 = MMDDriftDetector._sq_dists(X, X)
        tri = d2[np.triu_indices_from(d2, k=1)]
        tri = tri[np.isfinite(tri)]
        tri = tri[tri > 0.0]
        if tri.size == 0:
            return 1.0
        # Median heuristic: sigma = median(||x-y||)
        sigma = float(np.median(np.sqrt(tri)))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            return 1.0
        return sigma

    @staticmethod
    def _mmd2_from_kernel(K: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> float:
        idx1 = np.asarray(idx1, dtype=np.int64)
        idx2 = np.asarray(idx2, dtype=np.int64)
        n1 = int(idx1.size)
        n2 = int(idx2.size)
        if n1 < 2 or n2 < 2:
            return 0.0
        K11 = K[np.ix_(idx1, idx1)]
        K22 = K[np.ix_(idx2, idx2)]
        K12 = K[np.ix_(idx1, idx2)]
        term1 = float((np.sum(K11) - np.trace(K11)) / float(n1 * (n1 - 1)))
        term2 = float((np.sum(K22) - np.trace(K22)) / float(n2 * (n2 - 1)))
        term3 = float(np.mean(K12)) if (n1 * n2) > 0 else 0.0
        return float(max(0.0, term1 + term2 - 2.0 * term3))

    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        ref = np.asarray(reference_data, dtype=np.float64)
        cur = np.asarray(current_data, dtype=np.float64)
        if ref.ndim != 2:
            ref = ref.reshape(ref.shape[0], -1)
        if cur.ndim != 2:
            cur = cur.reshape(cur.shape[0], -1)

        n1 = int(ref.shape[0])
        n2 = int(cur.shape[0])
        p = int(min(ref.shape[1], cur.shape[1])) if ref.ndim == 2 and cur.ndim == 2 else 1
        ref = ref[:, :p]
        cur = cur[:, :p]

        if n1 < 2 or n2 < 2:
            # Not enough samples to estimate an unbiased MMD.
            return DriftDetectionResults(
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                change_points=[],
                drift_magnitude=0.0,
                statistical_significance={"sample_sizes": (n1, n2), "n_features": int(p), "reason": "insufficient_samples"},
                feature_drift_scores=np.zeros((p,), dtype=np.float64),
            )

        combined = np.vstack([ref, cur])
        sigma = self.bandwidth if self.bandwidth is not None else self._median_bandwidth(combined)
        sigma2 = float(max(1e-12, sigma * sigma))

        d2 = self._sq_dists(combined, combined)
        K = np.exp(-d2 / (2.0 * sigma2))

        idx1 = np.arange(n1, dtype=np.int64)
        idx2 = np.arange(n1, n1 + n2, dtype=np.int64)
        observed = self._mmd2_from_kernel(K, idx1, idx2)

        # Permutation test
        rng = np.random.default_rng(self.random_state)
        perm_stats = []
        for _ in range(int(self.n_permutations)):
            perm = rng.permutation(n1 + n2)
            perm_idx1 = perm[:n1]
            perm_idx2 = perm[n1:]
            perm_stats.append(self._mmd2_from_kernel(K, perm_idx1, perm_idx2))
        perm_arr = np.asarray(perm_stats, dtype=np.float64)
        p_value = float((1.0 + np.sum(perm_arr >= observed)) / (float(perm_arr.size) + 1.0))
        p_value = float(np.clip(p_value, 0.0, 1.0))
        drift_detected = p_value < float(self.alpha)

        critical_value = float(np.quantile(perm_arr, 1.0 - float(self.alpha))) if perm_arr.size else 0.0
        ci_lower = 0.0
        ci_upper = float(critical_value)

        # Feature-wise drift scores: standardized mean differences (same convention as Hotelling).
        feature_drift_scores = []
        for i in range(p):
            x1 = ref[:, i]
            x2 = cur[:, i]
            m1 = float(np.mean(x1))
            m2 = float(np.mean(x2))
            s1 = float(np.std(x1, ddof=1)) if x1.size >= 2 else 0.0
            s2 = float(np.std(x2, ddof=1)) if x2.size >= 2 else 0.0
            sp2 = (((x1.size - 1) * (s1 ** 2)) + ((x2.size - 1) * (s2 ** 2))) / float(max(1, x1.size + x2.size - 2))
            denom = float(np.sqrt(max(1e-12, sp2)))
            feature_drift_scores.append(abs(m1 - m2) / denom)
        feature_drift_scores = np.asarray(feature_drift_scores, dtype=np.float64)

        statistical_significance = {
            "mmd2": float(observed),
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "bandwidth": float(sigma),
            "n_permutations": int(perm_arr.size),
            "sample_sizes": (int(n1), int(n2)),
            "n_features": int(p),
        }

        return DriftDetectionResults(
            drift_detected=bool(drift_detected),
            drift_score=float(observed),
            p_value=float(p_value),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            change_points=[],
            drift_magnitude=float(observed),
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores,
        )


class KSDADCvMDriftDetector(BaseDriftDetector):
    """
    Distribution drift detector using a trio of classical two-sample tests:
      - KS (Kolmogorov-Smirnov)
      - AD (Anderson-Darling k-sample; here k=2)
      - CvM (Cramér–von Mises)

    Aggregation uses Bonferroni correction over the test family to control the
    family-wise error rate (FWER) at alpha = 1 - confidence_level.

    For multivariate inputs, we project to 1D (default: first principal component)
    to run the univariate tests in a well-defined way.
    """

    def __init__(self, confidence_level: float = 0.95, projection: str = "pca1"):
        super().__init__(confidence_level)
        self.projection = str(projection).lower()

    @staticmethod
    def _to_1d(x: np.ndarray, y: np.ndarray, *, projection: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        meta: Dict[str, object] = {
            "projection": projection,
            "input_shapes": (tuple(x.shape), tuple(y.shape)),
        }
        if x.ndim == 1 and y.ndim == 1:
            return x, y, meta
        if x.ndim != 2 or y.ndim != 2:
            return x.reshape(-1), y.reshape(-1), {**meta, "projection": "flatten"}

        # 2D -> 1D projection
        if x.shape[1] == 1 and y.shape[1] == 1:
            return x[:, 0], y[:, 0], {**meta, "projection": "feature0"}

        if projection in {"pca", "pca1", "pc1"}:
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=1, random_state=42)
                x1 = pca.fit_transform(x).reshape(-1)
                x2 = pca.transform(y).reshape(-1)
                meta["explained_variance_ratio"] = float(pca.explained_variance_ratio_[0])
                return x1, x2, meta
            except Exception as exc:
                meta["projection_error"] = str(exc)
                return x[:, 0], y[:, 0], {**meta, "projection": "feature0_fallback"}

        # Default fallback: feature 0.
        return x[:, 0], y[:, 0], {**meta, "projection": "feature0"}

    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        logger.info(
            "KS/AD/CvM drift detection: ref=%s cur=%s",
            getattr(reference_data, "shape", None),
            getattr(current_data, "shape", None),
        )

        x1, x2, proj_meta = self._to_1d(reference_data, current_data, projection=self.projection)
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        x1 = x1[np.isfinite(x1)]
        x2 = x2[np.isfinite(x2)]

        n1 = int(x1.size)
        n2 = int(x2.size)
        n_features = int(reference_data.shape[1]) if np.asarray(reference_data).ndim == 2 else 1

        # Guard: if too few samples, do not attempt to produce fragile p-values.
        if n1 < 2 or n2 < 2:
            stats_block = {
                "sample_sizes": (n1, n2),
                "n_features": n_features,
                "projection": proj_meta,
                "bonferroni_m": 3,
                "alpha": float(self.alpha),
                "reason": "insufficient_samples",
            }
            return DriftDetectionResults(
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 1.0),
                change_points=[],
                drift_magnitude=0.0,
                statistical_significance=stats_block,
                feature_drift_scores=np.zeros((n_features,), dtype=np.float64),
            )

        raw: Dict[str, Dict[str, float]] = {}
        pvals: List[float] = []

        # KS two-sample
        try:
            ks_res = stats.ks_2samp(x1, x2)
            ks_stat = float(getattr(ks_res, "statistic", ks_res[0]))
            ks_p = float(getattr(ks_res, "pvalue", ks_res[1]))
        except Exception:
            ks_stat, ks_p = 0.0, 1.0
        raw["ks"] = {"statistic": ks_stat, "p_value": float(np.clip(ks_p, 0.0, 1.0))}
        pvals.append(raw["ks"]["p_value"])

        # Anderson-Darling k-sample (k=2)
        ad_warnings: List[str] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ad_res = stats.anderson_ksamp([x1, x2])
            for w in caught:
                try:
                    ad_warnings.append(str(w.message))
                except Exception:
                    continue
            ad_stat = float(getattr(ad_res, "statistic", 0.0))
            # SciPy returns .pvalue in modern versions; .significance_level exists for compatibility.
            ad_p = getattr(ad_res, "pvalue", None)
            if ad_p is None:
                ad_p = getattr(ad_res, "significance_level", 1.0)
            ad_p = float(ad_p)
        except Exception:
            ad_stat, ad_p = 0.0, 1.0
        raw["ad"] = {"statistic": ad_stat, "p_value": float(np.clip(ad_p, 0.0, 1.0))}
        if ad_warnings:
            # Keep runs/tests clean while still preserving evidence for auditability.
            raw["ad"]["warnings"] = list(ad_warnings)
        pvals.append(raw["ad"]["p_value"])

        # Cramér–von Mises two-sample
        try:
            cvm_res = stats.cramervonmises_2samp(x1, x2)
            cvm_stat = float(getattr(cvm_res, "statistic", 0.0))
            cvm_p = float(getattr(cvm_res, "pvalue", 1.0))
        except Exception:
            cvm_stat, cvm_p = 0.0, 1.0
        raw["cvm"] = {"statistic": cvm_stat, "p_value": float(np.clip(cvm_p, 0.0, 1.0))}
        pvals.append(raw["cvm"]["p_value"])

        m = len(pvals)
        p_adj = _bonferroni_adjust(pvals, m=m)
        for key, p_a in zip(("ks", "ad", "cvm"), p_adj):
            raw[key]["p_value_bonferroni"] = float(p_a)

        p_combined = float(min(p_adj)) if p_adj else 1.0
        drift_detected = bool(p_combined < float(self.alpha))
        drift_score = float(np.clip(1.0 - p_combined, 0.0, 1.0))

        # Per-feature KS statistics are still useful for localization.
        try:
            ref2 = np.asarray(reference_data, dtype=np.float64)
            cur2 = np.asarray(current_data, dtype=np.float64)
            if ref2.ndim == 2 and cur2.ndim == 2 and ref2.shape[1] == cur2.shape[1]:
                feat_scores = []
                for j in range(ref2.shape[1]):
                    a = ref2[:, j]
                    b = cur2[:, j]
                    a = a[np.isfinite(a)]
                    b = b[np.isfinite(b)]
                    if a.size < 2 or b.size < 2:
                        feat_scores.append(0.0)
                    else:
                        feat_scores.append(float(stats.ks_2samp(a, b).statistic))
                feature_drift_scores = np.asarray(feat_scores, dtype=np.float64)
            else:
                feature_drift_scores = np.zeros((n_features,), dtype=np.float64)
        except Exception:
            feature_drift_scores = np.zeros((n_features,), dtype=np.float64)

        statistical_significance: Dict[str, object] = {
            "sample_sizes": (n1, n2),
            "n_features": n_features,
            "projection": proj_meta,
            "alpha": float(self.alpha),
            "bonferroni_m": int(m),
            "tests": raw,
            "p_value_bonferroni_min": float(p_combined),
        }

        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=float(p_combined),
            confidence_interval=(0.0, 1.0),
            change_points=[],
            drift_magnitude=drift_score,
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores,
        )


class PerDimKSDADCvMFisherBHDriftDetector(BaseDriftDetector):
    """
    Draft-parity statistical drift detector:
      - per-dimension KS/AD/CvM
      - Fisher aggregation across the 3 tests (per dimension)
      - BH/FDR correction across dimensions

    This is intentionally more "paper-like" than the PCA1+Bonferroni detector,
    but can be more expensive for high-dimensional features.
    """

    def __init__(self, confidence_level: float = 0.95, *, report_top_k: int = 25):
        super().__init__(confidence_level)
        self.report_top_k = int(max(0, report_top_k))

    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        ref = np.asarray(reference_data, dtype=np.float64)
        cur = np.asarray(current_data, dtype=np.float64)
        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)
        if ref.ndim != 2 or cur.ndim != 2:
            ref = ref.reshape(int(ref.shape[0]), -1)
            cur = cur.reshape(int(cur.shape[0]), -1)

        p = int(min(ref.shape[1], cur.shape[1]))
        ref = ref[:, :p]
        cur = cur[:, :p]

        # Sanitize non-finite values deterministically (rare, but happens in feature extraction).
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
        cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

        n1 = int(ref.shape[0])
        n2 = int(cur.shape[0])

        if n1 < 2 or n2 < 2 or p <= 0:
            stats_block = {
                "sample_sizes": (n1, n2),
                "n_features": int(p),
                "alpha": float(self.alpha),
                "aggregation": "fisher",
                "correction": "bh_fdr",
                "reason": "insufficient_samples",
            }
            return DriftDetectionResults(
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 1.0),
                change_points=[],
                drift_magnitude=0.0,
                statistical_significance=stats_block,
                feature_drift_scores=np.zeros((max(1, p),), dtype=np.float64),
            )

        ks_stats = np.zeros((p,), dtype=np.float64)
        p_ks = np.ones((p,), dtype=np.float64)
        p_ad = np.ones((p,), dtype=np.float64)
        p_cvm = np.ones((p,), dtype=np.float64)
        fisher_stat = np.zeros((p,), dtype=np.float64)
        fisher_p = np.ones((p,), dtype=np.float64)

        ad_warning_count = 0

        for j in range(p):
            x1 = ref[:, j]
            x2 = cur[:, j]
            x1 = x1[np.isfinite(x1)]
            x2 = x2[np.isfinite(x2)]
            if int(x1.size) < 2 or int(x2.size) < 2:
                continue

            # KS
            try:
                ks_res = stats.ks_2samp(x1, x2)
                ks_stats[j] = float(getattr(ks_res, "statistic", ks_res[0]))
                p_ks[j] = float(np.clip(float(getattr(ks_res, "pvalue", ks_res[1])), 0.0, 1.0))
            except Exception:
                ks_stats[j] = 0.0
                p_ks[j] = 1.0

            # AD (k-sample, k=2)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    ad_res = stats.anderson_ksamp([x1, x2])
                if caught:
                    ad_warning_count += int(len(caught))
                ad_p = getattr(ad_res, "pvalue", None)
                if ad_p is None:
                    ad_p = getattr(ad_res, "significance_level", 1.0)
                p_ad[j] = float(np.clip(float(ad_p), 0.0, 1.0))
            except Exception:
                p_ad[j] = 1.0

            # CvM
            try:
                cvm_res = stats.cramervonmises_2samp(x1, x2)
                p_cvm[j] = float(np.clip(float(getattr(cvm_res, "pvalue", 1.0)), 0.0, 1.0))
            except Exception:
                p_cvm[j] = 1.0

            stat_j, p_j = _fisher_combine_p_values([float(p_ks[j]), float(p_ad[j]), float(p_cvm[j])])
            fisher_stat[j] = float(stat_j)
            fisher_p[j] = float(p_j)

        fisher_q = np.asarray(_bh_fdr_adjust([float(x) for x in fisher_p.tolist()]), dtype=np.float64)
        p_min = float(np.min(fisher_q)) if fisher_q.size else 1.0
        drift_detected = bool(p_min < float(self.alpha))
        drift_score = float(np.clip(1.0 - p_min, 0.0, 1.0))

        # Top-K reporting by BH q-value.
        top_k = int(min(max(0, self.report_top_k), p))
        top_idx = np.argsort(fisher_q)[:top_k] if top_k > 0 else np.asarray([], dtype=int)
        top_features = []
        for idx in top_idx.tolist():
            j = int(idx)
            top_features.append(
                {
                    "feature_index": j,
                    "ks_statistic": float(ks_stats[j]),
                    "p_ks": float(p_ks[j]),
                    "p_ad": float(p_ad[j]),
                    "p_cvm": float(p_cvm[j]),
                    "fisher_statistic": float(fisher_stat[j]),
                    "fisher_p": float(fisher_p[j]),
                    "fisher_q_bh": float(fisher_q[j]),
                }
            )

        statistical_significance = {
            "sample_sizes": (int(n1), int(n2)),
            "n_features": int(p),
            "alpha": float(self.alpha),
            "aggregation": "fisher",
            "correction": "bh_fdr",
            "per_dimension": {
                "top_k": int(top_k),
                "top_features": top_features,
                "fisher_q_min": float(p_min),
                "ad_warning_count": int(ad_warning_count),
            },
        }

        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=float(p_min),
            confidence_interval=(0.0, 1.0),
            change_points=[],
            drift_magnitude=drift_score,
            statistical_significance=statistical_significance,
            feature_drift_scores=np.asarray(ks_stats, dtype=np.float64),
        )


class BayesianChangePointDetector(BaseDriftDetector):
    """
    Bayesian change point detection with posterior probabilities
    """
    
    def __init__(self, confidence_level: float = 0.95, prior_strength: float = 1.0):
        super().__init__(confidence_level)
        self.prior_strength = prior_strength
        
    def compute_posterior_probabilities(self, data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Compute posterior probabilities of change points
        
        Args:
            data: Time series data
            
        Returns:
            Posterior probabilities and detected change points
        """
        n_samples, n_features = data.shape

        posterior_probs = np.zeros(n_samples, dtype=np.float64)

        # Sliding window approach for change point detection
        window_size = min(50, max(2, n_samples // 4))
        if n_samples < (2 * window_size + 1):
            return posterior_probs, []

        # Work in log-space for numerical stability; final normalization is a softmax.
        log_scores = np.full(n_samples, -np.inf, dtype=np.float64)
        log_prior = -np.log(n_samples)

        def log_likelihood(data_points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
            diff = data_points - mean
            cov_inv = inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                return float("-inf")

            quad = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            return float(-0.5 * (np.sum(quad) + diff.shape[0] * (log_det + n_features * np.log(2.0 * np.pi))))

        for t in range(window_size, n_samples - window_size):
            before = data[t - window_size : t]
            after = data[t : t + window_size]

            try:
                mean_before = np.mean(before, axis=0)
                cov_before = np.cov(before, rowvar=False) + np.eye(n_features) * 1e-6

                mean_after = np.mean(after, axis=0)
                cov_after = np.cov(after, rowvar=False) + np.eye(n_features) * 1e-6

                combined = np.vstack([before, after])
                mean_combined = np.mean(combined, axis=0)
                cov_combined = np.cov(combined, rowvar=False) + np.eye(n_features) * 1e-6

                ll_before = log_likelihood(before, mean_before, cov_before)
                ll_after = log_likelihood(after, mean_after, cov_after)
                ll_change = ll_before + ll_after

                ll_null = log_likelihood(combined, mean_combined, cov_combined)
                log_bayes_factor = ll_change - ll_null

                log_scores[t] = float(self.prior_strength) * float(log_bayes_factor) + log_prior
            except Exception:
                # Fallback to simple variance-based score (still produces a reasonable peak score).
                var_before = np.var(before, axis=0)
                var_after = np.var(after, axis=0)
                var_ratio = float(np.mean(var_after / (var_before + 1e-8)))
                log_scores[t] = float(self.prior_strength) * float(abs(np.log(var_ratio))) + log_prior

        finite = np.isfinite(log_scores)
        if not np.any(finite):
            return posterior_probs, []

        max_log = float(np.max(log_scores[finite]))
        weights = np.zeros(n_samples, dtype=np.float64)
        weights[finite] = np.exp(log_scores[finite] - max_log)

        total = float(np.sum(weights))
        if total > 0:
            posterior_probs = weights / total

        threshold = float(np.percentile(posterior_probs, 95))
        change_points = np.where(posterior_probs > threshold)[0].tolist()
        return posterior_probs, change_points
    
    def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray) -> DriftDetectionResults:
        """
        Detect drift using Bayesian change point detection
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            DriftDetectionResults object
        """
        logger.info(f"Bayesian change point detection: ref={reference_data.shape}, cur={current_data.shape}")
        
        # Combine data for sequential analysis
        combined_data = np.vstack([reference_data, current_data])
        
        # Compute posterior probabilities
        posterior_probs, change_points = self.compute_posterior_probabilities(combined_data)
        
        # Check if change points occur in the transition region
        transition_start = max(0, len(reference_data) - 10)
        transition_end = min(len(combined_data) - 1, len(reference_data) + 10)
        
        transition_change_points = [
            cp for cp in change_points 
            if transition_start <= cp <= transition_end
        ]
        
        # Drift detected if change points found in transition region
        drift_detected = len(transition_change_points) > 0
        
        # Drift score as maximum posterior probability in transition region
        if drift_detected:
            transition_probs = posterior_probs[transition_start:transition_end + 1]
            drift_score = float(np.max(transition_probs))
        else:
            drift_score = 0.0
        
        # Approximate p-value based on posterior probability
        p_value = 1 - drift_score if drift_detected else 1.0
        
        # Feature-wise analysis (simplified)
        feature_drift_scores = []
        for i in range(reference_data.shape[1]):
            ref_mean = np.mean(reference_data[:, i])
            cur_mean = np.mean(current_data[:, i])
            ref_std = np.std(reference_data[:, i])
            cur_std = np.std(current_data[:, i])
            
            # Normalized difference
            mean_diff = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
            std_ratio = abs(np.log((cur_std + 1e-8) / (ref_std + 1e-8)))
            
            feature_drift_scores.append(mean_diff + std_ratio)
        feature_drift_scores = np.array(feature_drift_scores)
        
        # Acceptance region under the null (heuristic): posterior mass above the 95th percentile
        # is considered a change-point candidate.
        threshold = float(np.percentile(posterior_probs, 95))
        ci_lower = 0.0
        ci_upper = threshold
        
        # Statistical significance measures
        statistical_significance = {
            'max_posterior_prob': drift_score,
            'n_change_points': len(change_points),
            'transition_change_points': len(transition_change_points),
            'mean_posterior_prob': float(np.mean(posterior_probs)),
            'cp_threshold_p95': threshold,
            'transition_window': (transition_start, transition_end),
            'sample_sizes': (int(reference_data.shape[0]), int(current_data.shape[0])),
            'n_features': int(reference_data.shape[1]) if reference_data.ndim == 2 else 1,
        }
        
        return DriftDetectionResults(
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            change_points=change_points,
            drift_magnitude=drift_score,
            statistical_significance=statistical_significance,
            feature_drift_scores=feature_drift_scores
        )

class EnhancedDriftDetector:
    """
    Enhanced drift detector combining multiple methods
    """
    
    def __init__(self, confidence_level: float = 0.95, methods: List[str] = None):
        """
        Initialize enhanced drift detector
        
        Args:
            confidence_level: Statistical confidence level
            methods: List of methods to use ['hotelling', 'ks', 'bayesian']
        """
        self.confidence_level = confidence_level
        
        if methods is None:
            methods = ['hotelling', 'ks', 'bayesian']
        
        self.detectors = {}
        
        if 'hotelling' in methods:
            self.detectors['hotelling'] = HotellingT2DriftDetector(confidence_level)
        if 'ks' in methods:
            self.detectors['ks'] = KolmogorovSmirnovDriftDetector(confidence_level)
        if 'mmd' in methods:
            self.detectors['mmd'] = MMDDriftDetector(confidence_level)
        if 'ks_ad_cvm' in methods or 'ksadcvm' in methods:
            self.detectors['ks_ad_cvm'] = KSDADCvMDriftDetector(confidence_level)
        if (
            'ks_ad_cvm_fisher_bh' in methods
            or 'ks_ad_cvm_fisherbh' in methods
            or 'ks_ad_cvm_per_dim' in methods
            or 'ksadcvm_fisher_bh' in methods
        ):
            self.detectors['ks_ad_cvm_fisher_bh'] = PerDimKSDADCvMFisherBHDriftDetector(confidence_level)
        if 'bayesian' in methods:
            self.detectors['bayesian'] = BayesianChangePointDetector(confidence_level)
    
    def detect_drift_ensemble(
        self, 
        reference_data: np.ndarray, 
        current_data: np.ndarray
    ) -> Dict[str, DriftDetectionResults]:
        """
        Detect drift using ensemble of methods
        
        Args:
            reference_data: Reference data matrix
            current_data: Current data matrix
            
        Returns:
            Dictionary of results from each method
        """
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                result = detector.detect_drift(reference_data, current_data)
                results[name] = result
                logger.info(f"{name} drift detection: {result.drift_detected} (score: {result.drift_score:.4f})")
            except Exception as e:
                logger.error(f"Error in {name} drift detection: {e}")
                # Create dummy result
                results[name] = DriftDetectionResults(
                    drift_detected=False,
                    drift_score=0.0,
                    p_value=1.0,
                    confidence_interval=(0.0, 0.0),
                    change_points=[],
                    drift_magnitude=0.0,
                    statistical_significance={},
                    feature_drift_scores=np.zeros(reference_data.shape[1])
                )
        
        return results
    
    def get_consensus_result(self, results: Dict[str, DriftDetectionResults]) -> DriftDetectionResults:
        """
        Get consensus result from multiple detectors
        
        Args:
            results: Dictionary of results from different methods
            
        Returns:
            Consensus drift detection result
        """
        # Count number of methods detecting drift
        drift_votes = sum(1 for result in results.values() if result.drift_detected)
        total_methods = len(results)
        
        # Consensus: drift detected if majority of methods agree
        consensus_drift = drift_votes > total_methods / 2
        
        # Combine evidence via p-values (more comparable than raw drift scores across methods).
        p_values: List[float] = []
        for r in results.values():
            p = float(r.p_value)
            if not np.isfinite(p):
                p = 1.0
            p_values.append(float(np.clip(p, 0.0, 1.0)))

        consensus_p_value = float(np.mean(p_values)) if p_values else 1.0
        consensus_score = float(np.mean([1.0 - p for p in p_values])) if p_values else 0.0

        # Consensus score lives in [0,1] by construction.
        consensus_ci = (0.0, 1.0)
        
        # Combine change points
        all_change_points = []
        for result in results.values():
            all_change_points.extend(result.change_points)
        unique_change_points = sorted(list(set(all_change_points)))
        
        # Average feature drift scores
        feature_scores = np.array([result.feature_drift_scores for result in results.values()])
        consensus_feature_scores = np.mean(feature_scores, axis=0)
        
        # Combined statistical significance
        sample_sizes = None
        n_features = None
        for r in results.values():
            try:
                ss = (r.statistical_significance or {}).get("sample_sizes")
                if ss is not None:
                    sample_sizes = ss
                nf = (r.statistical_significance or {}).get("n_features")
                if nf is not None:
                    n_features = nf
            except Exception:
                continue
            if sample_sizes is not None and n_features is not None:
                break

        consensus_stats = {
            'drift_votes': drift_votes,
            'total_methods': total_methods,
            'consensus_strength': drift_votes / total_methods,
            'individual_results': {name: result.statistical_significance for name, result in results.items()},
            'sample_sizes': sample_sizes,
            'n_features': n_features,
        }
        
        return DriftDetectionResults(
            drift_detected=consensus_drift,
            drift_score=consensus_score,
            p_value=consensus_p_value,
            confidence_interval=consensus_ci,
            change_points=unique_change_points,
            drift_magnitude=consensus_score,
            statistical_significance=consensus_stats,
            feature_drift_scores=consensus_feature_scores
        )


def create_enhanced_drift_detector(
    confidence_level: float = 0.95,
    methods: List[str] = None
) -> EnhancedDriftDetector:
    """
    Factory function to create enhanced drift detector
    
    Args:
        confidence_level: Statistical confidence level
        methods: List of detection methods to use
        
    Returns:
        Configured drift detector
    """
    return EnhancedDriftDetector(confidence_level, methods)


# Example usage and testing
def demonstrate_enhanced_drift_detection():
    """Demonstrate enhanced drift detection capabilities"""
    print("=== NeurInSpectre Enhanced Drift Detection Demo ===")
    
    # Generate sample data with drift
    np.random.seed(42)
    n_samples, n_features = 500, 4
    
    # Reference data (stable)
    reference_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_samples
    )
    
    # Current data with drift (shifted mean and different covariance)
    drift_mean = np.array([1.5, -1.0, 0.5, -0.8])
    drift_cov = np.array([
        [1.5, 0.3, 0.1, 0.2],
        [0.3, 2.0, 0.2, 0.1],
        [0.1, 0.2, 0.8, 0.3],
        [0.2, 0.1, 0.3, 1.2]
    ])
    
    current_data = np.random.multivariate_normal(
        mean=drift_mean,
        cov=drift_cov,
        size=n_samples
    )
    
    print(f"Reference data: {reference_data.shape}")
    print(f"Current data: {current_data.shape}")
    print(f"True drift: Mean shift = {drift_mean}")
    
    # Create enhanced drift detector
    detector = create_enhanced_drift_detector(
        confidence_level=0.95,
        methods=['hotelling', 'ks', 'bayesian']
    )
    
    # Detect drift using all methods
    results = detector.detect_drift_ensemble(reference_data, current_data)
    
    # Get consensus result
    consensus = detector.get_consensus_result(results)
    
    print("\n=== Individual Method Results ===")
    for method, result in results.items():
        print(f"{method.upper()}:")
        print(f"  Drift Detected: {result.drift_detected}")
        print(f"  Drift Score: {result.drift_score:.4f}")
        print(f"  P-value: {result.p_value:.6f}")
        print(f"  Confidence Interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        print(f"  Feature Drift Scores: {result.feature_drift_scores}")
        print()
    
    print("=== Consensus Result ===")
    print(f"Consensus Drift Detected: {consensus.drift_detected}")
    print(f"Consensus Score: {consensus.drift_score:.4f}")
    print(f"Consensus P-value: {consensus.p_value:.6f}")
    print(f"Consensus Strength: {consensus.statistical_significance['consensus_strength']:.3f}")
    print(f"Feature Drift Scores: {consensus.feature_drift_scores}")
    
    if consensus.change_points:
        print(f"Detected Change Points: {consensus.change_points}")


if __name__ == "__main__":
    demonstrate_enhanced_drift_detection() 