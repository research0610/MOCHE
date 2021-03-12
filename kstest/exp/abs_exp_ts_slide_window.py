import random
import tqdm
from kstest.baselines import SpectralResidual, Luminol
from kstest.shift_detector import ShiftDetectorMSP
from kstest.utils import set_random_seed
from kstest.log import getLogger
import numpy as np
from kstest.dataset.ts_data import TSData, ReadNABDatasetWithTime

set_random_seed()

logger = getLogger(__name__)


class AbsExpTS:

    SR = "SR"
    SR_MOCHI = "SR_MOCHI"
    SR_MOCHI_NL = "SR_MOCHI_NL"
    SR_CORNER_SEARCH = "SR_CORNER_SEARCH"
    SR_GRACE = "SR_GRACE"
    SR_GREEDY = "SR_GREEDY"
    SR_GRACE_ORIGIN = "SR_GRACE_ORIGIN"
    SR_CORNER_SEARCH_ORIGIN = "SR_CORNER_SEARCH_ORIGIN"

    L = "L"
    L_MOCHI = "L_MOCHI"
    L_MOCHI_NL = "L_MOCHI_NL"
    L_CORNER_SEARCH = "L_CORNER_SEARCH"
    L_CORNER_SEARCH_ORIGIN = "L_CORNER_SEARCH_ORIGIN"
    L_GRACE = "L_GRACE"
    L_GRACE_ORIGIN = "L_GRACE_ORIGIN"
    L_GREEDY = "L_GREEDY"

    STOMP = "STOMP"
    STOMP_005 = "STOMP_0.05"
    S2G = "S2G"
    S2G_005 = "S2G_0.05"
    DENSITY = "DENSITY" #D3 baseline


    def get_interpretation(self, X_tr_red, X_te_red, X_tr, X_te, data_file, ground_truth: set):
        if self.method == self.SR:
            sr = SpectralResidual(X_tr_red, X_te_red, self.shift_detector, self.level)
            return {
                self.SR_MOCHI: sr.interpret_by_mci(ground_truth),
                self.SR_MOCHI_NL: sr.interpret_by_mci_no_bound(),
                self.SR_CORNER_SEARCH: sr.interpret_by_corner_search(ground_truth,  k_max=100),
                self.SR_GREEDY: sr.interpret_by_greedy_metric(ground_truth),
                self.SR_GRACE: sr.interpret_by_grace(ground_truth, k_max=100),
                self.STOMP_005: sr.interpret_by_stomp(ground_truth, 0.05),
                self.S2G_005: sr.interpret_by_s2g(ground_truth, 0.05),
                self.DENSITY: sr.interpret_by_density(ground_truth)
            }
        elif self.method == self.L:
            l = Luminol(X_tr_red, X_te_red, self.shift_detector, self.level)
            return {
                self.L_MOCHI: l.interpret_by_mci(ground_truth),
                self.L_MOCHI_NL: l.interpret_by_mci_no_bound(),
                self.L_CORNER_SEARCH: l.interpret_by_corner_search(ground_truth,  k_max=100),
                self.L_GREEDY: l.interpret_by_greedy_metric(ground_truth),
                self.L_GRACE: l.interpret_by_grace(ground_truth,  k_max=100),
                self.STOMP_005: l.interpret_by_stomp(ground_truth, 0.05),
                self.S2G_005: l.interpret_by_s2g(ground_truth, 0.05),
                self.DENSITY: l.interpret_by_density(ground_truth)
            }
        else:
            raise Exception(f"Explainer {self.method} is not supported")

    def __init__(self, dataset, method, level, sample_size):
        set_random_seed()
        self.dataset = dataset
        self.method = method
        self.level = level
        # self.ts = load_ts_dataset(dataset)
        for e in TSData:
            if dataset == e.name:
                self.ts = ReadNABDatasetWithTime(e)
        self.sample_size = sample_size
        logger.info(f"Data Name {dataset}")

        self.shift_detector = ShiftDetectorMSP(self.level, self.dataset, None)

    def find_failed_ks_tests(self, window_size):
        # ============================
        # Generate drifted data
        # ============================
        total_files = len(self.ts)
        file_counter = 0
        for file, (origin_ts, labels, times) in self.ts.items():
            set_random_seed()
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

                reference_time = times[reference_start: reference_end]
                test_time = times[test_start: test_end]

                if window_size > sum(test_label) > 0 and sum(reference_label) == 0:
                    if self.check_drift(test_set, reference_set):
                        generated_data.append((test_set, reference_set, test_label.tolist(), file,
                                               reference_time, test_time))
            if len(generated_data) > 0:
                ks_test_data = [random.choice(generated_data) for _ in range(self.sample_size)]
                for i in ks_test_data:
                    logger.info(f"Processing files {file_counter}/{total_files}")
                    yield i


    def check_drift(self, X_te_red, X_tr_red):
        X_tr_odim = (-np.amax(X_tr_red, axis=1)).tolist()
        X_te_odim = (-np.amax(X_te_red, axis=1)).tolist()
        return self.shift_detector.detect_by_msp(X_tr_odim, X_te_odim)
