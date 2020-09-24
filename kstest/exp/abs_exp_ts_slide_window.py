import random

import tqdm

from kstest.baselines import SpectralResidual, Luminol
from kstest.shift_detector import ShiftDetectorMSP
from kstest.utils import set_random_seed
from kstest.log import getLogger
import numpy as np
from kstest.dataset.ts_data import load_ts_dataset

set_random_seed()

logger = getLogger(__name__)


class AbsExpTS:

    SR = "SR"
    SR_MOCHI = "SR_MOCHI"
    SR_MOCHI_NL = "SR_MOCHI_NL"
    SR_MOCHI_NP = "SR_MOCHI_NP"
    SR_CORNER_SEARCH = "SR_CORNER_SEARCH"
    SR_GRACE = "SR_GRACE"
    SR_GREEDY = "SR_GREEDY"
    SR_GRACE_ORIGIN = "SR_GRACE_ORIGIN"
    SR_CORNER_SEARCH_ORIGIN = "SR_CORNER_SEARCH_ORIGIN"

    L = "L"
    L_MOCHI = "L_MOCHI"
    L_MOCHI_NL = "L_MOCHI_NL"
    L_MOCHI_NP = "L_MOCHI_NP"
    L_CORNER_SEARCH = "L_CORNER_SEARCH"
    L_CORNER_SEARCH_ORIGIN = "L_CORNER_SEARCH_ORIGIN"
    L_GRACE = "L_GRACE"
    L_GRACE_ORIGIN = "L_GRACE_ORIGIN"
    L_GREEDY = "L_GREEDY"


    def get_interpretation(self, X_tr_red, X_te_red, X_tr, X_te, data_file, ground_truth: set):
        if self.method == self.SR:
            sr = SpectralResidual(X_tr_red, X_te_red, self.shift_detector, self.level)
            return {
                self.SR_MOCHI: sr.interpret_by_mci(ground_truth),
                self.SR_MOCHI_NL: sr.interpret_by_mci_no_bound(),
                self.SR_MOCHI_NP: sr.interpret_by_mci_no_prune(),
                self.SR_CORNER_SEARCH: sr.interpret_by_corner_search(ground_truth,  k_max=100),
                self.SR_GREEDY: sr.interpret_by_greedy_metric(ground_truth),
                self.SR_GRACE: sr.interpret_by_grace(ground_truth, k_max=100),
            }
        elif self.method == self.L:
            l = Luminol(X_tr_red, X_te_red, self.shift_detector, self.level)
            return {
                self.L_MOCHI: l.interpret_by_mci(ground_truth),
                self.L_MOCHI_NL: l.interpret_by_mci_no_bound(),
                self.L_MOCHI_NP: l.interpret_by_mci_no_prune(),
                self.L_CORNER_SEARCH: l.interpret_by_corner_search(ground_truth,  k_max=100),
                self.L_GREEDY: l.interpret_by_greedy_metric(ground_truth),
                self.L_GRACE: l.interpret_by_grace(ground_truth,  k_max=100),
            }
        else:
            raise Exception(f"Explainer {self.method} is not supported")

    def __init__(self, dataset, method, level, sample_size):
        set_random_seed()
        self.dataset = dataset
        self.method = method
        self.level = level
        self.ts = load_ts_dataset(dataset)
        self.sample_size = sample_size
        logger.info(f"Data Name {dataset}")

        self.shift_detector = ShiftDetectorMSP(self.level, self.dataset, None)

    def find_failed_ks_tests(self, window_size):
        # ============================
        # Generate drifted data
        # ============================
        total_files = len(self.ts)
        file_counter = 0
        for file, (origin_ts, labels) in self.ts.items():
            file_counter += 1
            _labels = labels.squeeze().tolist()
            logger.info(f"Window size {window_size}")
            generated_data = []
            for reference_start in tqdm.trange(origin_ts.shape[0]):
                reference_end = reference_start + window_size
                test_start = reference_end
                test_end = test_start + window_size
                if test_end > origin_ts.shape[0]:
                    break
                reference_set = origin_ts[reference_start: reference_end, :]
                test_set = origin_ts[test_start: test_end, :]
                test_label = labels[test_start: test_end].squeeze()
                reference_label = labels[reference_start: reference_end]
                if window_size > sum(test_label) > 0 and sum(reference_label) == 0:
                    if self.check_drift(test_set, reference_set):
                        generated_data.append((test_set, reference_set, test_label.tolist(), file))
            if len(generated_data) > 0:
                for i in range(self.sample_size):
                    logger.info(f"Processing files {file_counter}/{total_files}")
                    yield random.choice(generated_data)


    def find_failed_ks_tests_bak(self, ratio):
        # ============================
        # Generate drifted data
        # ============================
        total_files = len(self.ts)
        file_counter = 0
        for file, (origin_ts, labels) in self.ts.items():
            file_counter += 1
            _labels = labels.squeeze().tolist()
            counters = []
            prev = 0
            counter = 0
            for i in _labels:
                if i == 1:
                    counter += 1
                elif i == 0 and prev == 1:
                    counters.append(counter)
                    counter = 0
                prev = i
            noise_size = max(counters)
            window_size = int(noise_size / ratio)
            logger.info(f"Noise size {noise_size} Window size {window_size}")
            generated_data = []
            for reference_start in tqdm.trange(origin_ts.shape[0]):
                reference_end = reference_start + window_size
                test_start = reference_end
                test_end = test_start + window_size
                if test_end > origin_ts.shape[0]:
                    break
                reference_set = origin_ts[reference_start: reference_end, :]
                test_set = origin_ts[test_start: test_end, :]
                test_label = labels[test_start: test_end].squeeze()
                reference_label = labels[reference_start: reference_end]
                if window_size > sum(test_label) > 0 and sum(reference_label) == 0:
                    if self.check_drift(test_set, reference_set):
                        generated_data.append((test_set, reference_set, test_label.tolist(), file))
            if len(generated_data) > 0:
                for i in range(10):
                    logger.info(f"Processing files {file_counter}/{total_files}")
                    yield random.choice(generated_data)


    def check_drift(self, X_te_red, X_tr_red):
        X_tr_odim = (-np.amax(X_tr_red, axis=1)).tolist()
        X_te_odim = (-np.amax(X_te_red, axis=1)).tolist()
        return self.shift_detector.detect_by_msp(X_tr_odim, X_te_odim)
