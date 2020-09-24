from luminol.anomaly_detector import AnomalyDetector
from kstest.corner_search import CornerSearch
from kstest.grace import Grace
from kstest.mochi import Interpreter
from kstest.ks_test import do_ks_test
from kstest.log import getLogger
import time

logger = getLogger(__name__)


def get_segments(ground_truth, x_te_length):
    labels = [1 if i in ground_truth else 0 for i in range(x_te_length)]
    _segs = set()
    abnormal_segments = set()
    prev = None
    for idx, i in enumerate(labels):
        if i == 1:
            _segs.add(idx)
        if i != prev and i == 0 and len(_segs) > 0:
            abnormal_segments.add(tuple(_segs))
            _segs = set()
        prev = i
    if len(_segs) > 0:
        abnormal_segments.add(tuple(_segs))
    assert len(abnormal_segments) > 0, (labels, abnormal_segments)
    return abnormal_segments


def compute_metric(interpretation, ground_truth, X_te_odim, X_tr_odim, level, scores):
    interpretation = set(interpretation)
    precision = len(interpretation & ground_truth) / len(interpretation) if len(interpretation) > 0 else 0
    recall = len(interpretation & ground_truth) / len(ground_truth) if len(ground_truth) > 0 else 0
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    idx_X_te = [X_te_odim[idx] for idx in range(len(X_te_odim)) if idx not in interpretation]

    # To handle the special case that the Grace method could remove all points
    if_shift = False
    if len(idx_X_te) > 0:
        if_shift = do_ks_test(X_tr_odim, idx_X_te, level)  # Return True if there is a distribution drift

    fpr = 0
    if len(X_te_odim) - len(ground_truth) > 0:
        fpr = len(interpretation - ground_truth) / (len(X_te_odim) - len(ground_truth))

    kdd_precision = 0
    kdd_recall = 0
    kdd_f1 = 0
    if precision != 0:
        segment_index = {}
        for segs in get_segments(ground_truth, len(X_te_odim)):
            for i in segs:
                segment_index[i] = set(segs)

        kdd_interpretation = set()
        for i in interpretation:
            if i in segment_index:
                kdd_interpretation |= segment_index[i]
        kdd_interpretation |= interpretation

        kdd_capture = kdd_interpretation & ground_truth
        kdd_precision = len(kdd_capture) / len(kdd_interpretation)
        kdd_recall = 1
        kdd_f1 = 2 * kdd_precision * kdd_recall / (kdd_precision + kdd_recall)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "KDDPrecision": kdd_precision,
        "KDDRecall": kdd_recall,
        "KDDF1": kdd_f1,
        "IFShift": 1 if if_shift else 0,
        "Ratio": len(ground_truth) / len(X_te_odim),
        "TPR": recall,
        "FPR": fpr,
        "SIZE": len(X_te_odim),
        "Interpretation": {i: scores[i] for i in interpretation}
    }


class AbsBaseLine:
    def __init__(self, X_tr_odim, X_te_odim, scores, shift_detector, level):
        # Large score means anomaly
        self.scores = scores
        self.X_tr_odim = X_tr_odim
        self.X_te_odim = X_te_odim
        self.shift_detector = shift_detector
        self.level = level

    def interpret_by_corner_search(self, ground_truth, k_max):
        X_te_sorted = sorted([(idx, self.scores[idx]) for idx, v in enumerate(self.X_te_odim)],
                             key=lambda i: i[1],
                             reverse=True)
        L = [idx for idx, kv in X_te_sorted]

        st = time.time()
        corner_search = CornerSearch(self.X_te_odim, self.X_tr_odim, self.level, k_max=k_max, ranked_list=L)
        interpretation = corner_search.generate_counterfactual()
        ed = time.time()

        ground_truth = set(ground_truth)
        rst = compute_metric(interpretation, ground_truth, self.X_te_odim, self.X_tr_odim, self.level,
                             self.scores)
        rst["Time"] = ed - st
        return rst

    def interpret_by_grace(self, ground_truth, k_max):
        X_te_sorted = sorted([(idx, self.scores[idx]) for idx, v in enumerate(self.X_te_odim)],
                             key=lambda i: i[1],
                             reverse=True)
        L = [idx for idx, kv in X_te_sorted]

        st = time.time()
        grace = Grace(self.X_te_odim, self.X_tr_odim, self.level, k_max=k_max, ranked_list=L)
        interpretation = grace.generate_counterfactual()
        ed = time.time()

        ground_truth = set(ground_truth)
        rst = compute_metric(interpretation, ground_truth, self.X_te_odim, self.X_tr_odim, self.level,
                             self.scores)
        rst["Time"] = ed - st
        return rst

    def interpret_by_mci(self, ground_truth):
        X_te_sorted = sorted([(idx, self.scores[idx]) for idx, v in enumerate(self.X_te_odim)],
                             key=lambda i: i[1],
                             reverse=True)
        X_te_rank = [idx for idx, kv in X_te_sorted]

        st = time.time()
        interpret = Interpreter(self.X_te_odim, self.X_tr_odim, X_te_rank, self.level)
        k, k_estimate = interpret.find_k()
        interpretation, ks_statistic = interpret.do_interpretation_given_k(k)
        ed = time.time()

        ground_truth = set(ground_truth)
        assert interpret.check_k(k, if_necessary_sufficient=True)
        rst = compute_metric(interpretation, ground_truth, self.X_te_odim, self.X_tr_odim,
                             self.level, self.scores)
        rst["Time"] = ed - st
        rst["EstimateError"]: k - k_estimate
        return rst

    def interpret_by_mci_no_bound(self):
        X_te_sorted = sorted([(idx, self.scores[idx]) for idx, v in enumerate(self.X_te_odim)],
                             key=lambda i: i[1],
                             reverse=True)
        X_te_rank = [idx for idx, kv in X_te_sorted]

        st = time.time()
        interpret = Interpreter(self.X_te_odim, self.X_tr_odim, X_te_rank, self.level)
        k = interpret.find_k_brute_force()
        interpretation, ks_statistic = interpret.do_interpretation_given_k(k)
        ed = time.time()
        return {"Time": ed - st, "Size": len(self.X_te_odim), "Level": self.level,
                "Interpretation": list(interpretation)}

    def interpret_by_mci_no_prune(self):
        X_te_sorted = sorted([(idx, self.scores[idx]) for idx, v in enumerate(self.X_te_odim)],
                             key=lambda i: i[1],
                             reverse=True)
        X_te_rank = [idx for idx, kv in X_te_sorted]

        st = time.time()
        interpret = Interpreter(self.X_te_odim, self.X_tr_odim, X_te_rank, self.level)
        k = interpret.find_k_brute_force()
        interpretation, ks_statistic = interpret.do_interpretation_given_k_no_prune(k)
        ed = time.time()
        return {"Time": ed - st, "Size": len(self.X_te_odim), "Level": self.level,
                "Interpretation": list(interpretation)}

    def interpret_by_greedy_metric(self, ground_truth):
        X_te_idx_sorted = list(sorted(range(len(self.X_te_odim)), key=lambda i: self.scores[i], reverse=True))

        st = time.time()
        for i in range(1, len(self.scores)):
            interpretation = X_te_idx_sorted[:i]
            _rst = compute_metric(interpretation, ground_truth, self.X_te_odim, self.X_tr_odim, self.level, self.scores)
            if _rst["IFShift"] == 0:
                break
        ed = time.time()
        return {"Time": ed - st, "Size": len(self.X_te_odim), "Level": self.level}


class SpectralResidual(AbsBaseLine):
    def __init__(self, X_tr_red, X_te_red, shift_detector, level):
        """
        Adopt the parameter settings from
        https://docs.seldon.io/projects/alibi-detect/en/latest/examples/od_sr_synth.html
        """
        from alibi_detect.od import SpectralResidual as SR
        logger.info("Run Spectral Residual")
        X_tr_odim = (-np.amax(X_tr_red, axis=1)).tolist()
        X_te_odim = (-np.amax(X_te_red, axis=1)).tolist()
        od = SR(
            threshold=0,  # threshold for outlier score
            window_amp=20,  # window for the average log amplitude 3
            window_local=20,  # window for the average saliency map 21
            n_est_points=20  # nb of estimated points padded to the end of the sequence 5
        )
        score = od.score(np.vstack([X_te_red, ]))[-len(X_te_red):]
        assert len(score) == len(X_te_odim), (len(score), len(X_te_odim))
        super().__init__(X_tr_odim, X_te_odim, score, shift_detector, level)


class Luminol(AbsBaseLine):
    def __init__(self, X_tr_red, X_te_red, shift_detector, level):
        logger.info("Run Luminol")
        X_tr_odim = (-np.amax(X_tr_red, axis=1)).tolist()
        X_te_odim = (-np.amax(X_te_red, axis=1)).tolist()
        ts = X_te_odim
        ts = {i: v for i, v in enumerate(ts)}
        train_ts = {i: v for i, v in enumerate(X_tr_odim)}
        my_detector = AnomalyDetector(ts, baseline_time_series=train_ts,
                                      algorithm_params={'precision': 10, 'lag_window_size': 0.1,
                                                        'future_window_size': 0.1, 'chunk_size': 2})
        _score = my_detector.get_all_scores()
        score = []
        for i in range(len(X_te_odim)):
            score.append(_score[i])
        assert len(score) == len(X_te_odim), (len(score), len(X_te_odim))
        super().__init__(X_tr_odim, X_te_odim, score, shift_detector, level)
