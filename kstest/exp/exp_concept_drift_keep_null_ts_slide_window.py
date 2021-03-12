import argparse
import json
from collections import defaultdict
from kstest.exp.abs_exp_ts_slide_window import AbsExpTS
import kstest.utils as utils
import numpy as np
from kstest.log import getLogger

logger = getLogger(__name__)


# ==========================================================================


class Experiment(AbsExpTS):
    def __init__(self, dataset, method):
        self.output_file = f"result/experiment_drift_keep_null_{dataset}_{method}_slide_window_new.json"
        self.data_segments = f"result/ts_segments_{dataset}_{method}_slide_window_new.json"
        self.window_size = [100, 200, 300, 500, 1000, 1500, 2000]
        super().__init__(dataset, method, 0.05, sample_size=10)

    @staticmethod
    def get_file_path(dataset, method):
        return f"result/experiment_drift_keep_null_{dataset}_{method}_slide_window_new.json"

    def run(self):
        total_f1 = defaultdict(list)
        total_true_f1 = defaultdict(list)
        total_size = defaultdict(list)
        total_shift = defaultdict(list)
        with open(self.data_segments, "w") as w, open(self.output_file, "w") as w_1:
            for window in self.window_size:
                # for X_te_red, X_tr_red, labels, file in self.find_failed_ks_tests(window):
                for X_te_red, X_tr_red, labels, file, reference_time, test_time in self.find_failed_ks_tests(window):
                    X_tr_odim = (np.amax(X_tr_red, axis=1)).tolist()
                    X_te_odim = (np.amax(X_te_red, axis=1)).tolist()
                    if_shift = self.shift_detector.detect_by_msp(X_tr_odim, X_te_odim)
                    logger.info(f"If shift: {if_shift} ground truth size {len(labels)} noise size {sum(labels)}")

                    if not if_shift:
                        continue

                    # # k -> interpretation
                    ground_truth = {idx for idx, label in enumerate(labels) if label == 1}
                    interpretations = self.get_interpretation(X_tr_red, X_te_red, X_tr_red, X_te_red, file,
                                                              ground_truth)
                    mochi_size = 0
                    for method, rst in interpretations.items():
                        if "MOCHI" in method and "MOCHI_NL" not in method:
                            mochi_size = len(rst["Interpretation"])
                            logger.info(f"Mochi size is {mochi_size}")
                    logger.info("")  # Create a new line
                    logger.info("=" * 50)
                    logger.info(f"Interpretation size {mochi_size}")
                    for method, info in interpretations.items():
                        print(f"{method} {info}")
                        if "MOCHI_NL" in method:
                            continue
                        if int(info["IFShift"]) == 0:
                            total_size[method].append(len(info["Interpretation"]))
                        total_f1[method].append(info["KDDF1"])
                        total_true_f1[method].append(info["F1"])
                        total_shift[method].append(1 if info["IFShift"] else 0)

                    for method, f1s in total_f1.items():
                        logger.info(
                            f"{method} {np.mean(f1s)} {np.mean(total_size[method])} {np.mean(total_true_f1[method])} {np.mean(total_shift[method])}")

                    w.write(json.dumps({
                        "REFERENCE": X_tr_odim,
                        "TEST": X_te_odim,
                        "LABELS": labels,
                        "FILE": file,
                        "REFERENCE_TS": reference_time,
                        "TEST_TS": test_time
                    }) + "\n")
                    w_1.write(json.dumps(interpretations) + "\n")
                    logger.info(self.output_file)
        logger.info(self.output_file)
        logger.info(self.data_segments)


def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Source Data", required=True)
    parser.add_argument("--explainer", help="name of explainer")
    parser.add_argument("--gpu", help="GPU used", choices=["cuda:1", "cuda:0", "cuda:2", "cuda:3", "cpu"])
    parsedArgs = parser.parse_args(sys.argv[1:])
    dataset = parsedArgs.dataset
    expln_name = parsedArgs.explainer
    utils.DEVICE = parsedArgs.gpu

    logger.info("=" * 50)
    logger.info(f"Start {dataset} {expln_name}")
    logger.info("=" * 50)

    gpu_names = {
        "cuda:0": '/device:GPU:0',
        "cuda:1": '/device:GPU:1',
        "cuda:2": '/device:GPU:2',
        "cuda:3": '/device:GPU:3',
    }

    exp = Experiment(dataset, expln_name)
    exp.run()

    logger.info("=" * 50)
    logger.info(f"Finish {dataset} {expln_name}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
