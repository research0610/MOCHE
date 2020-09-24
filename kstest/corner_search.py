from typing import List
import tqdm
from kstest.ks_test import do_ks_test
import numpy as np


class CornerSearch:
    def compute_max_cdf(self, test_list, reference_list):
        T = 0
        R = 1
        all_values = list(sorted(set(test_list + reference_list)))
        _test_set = [(T, v) for v in test_list]
        _reference_set = [(R, v) for v in reference_list]
        _all_values = sorted(_test_set + _reference_set, key=lambda i: i[1])
        t_counter, r_counter = 0, 0
        test_set_cdf = {}
        reference_set_cdf = {}
        for (value_source, value) in _all_values:
            if value_source == T:
                t_counter += 1
            else:
                r_counter += 1
            test_set_cdf[value] = t_counter / len(test_list)
            reference_set_cdf[value] = r_counter / len(reference_list)
        return max([abs(reference_set_cdf[v] - test_set_cdf[v]) for v in all_values])

    def compute_point_importance(self, test_list, reference_list):
        origin_cdf_diff = self.compute_max_cdf(test_list, reference_list)
        perturb_effects = {}
        for i in range(len(test_list)):
            _test_list = test_list[0:i] + test_list[i + 1:]
            perturbed_cdf_diff = self.compute_max_cdf(_test_list, reference_list)
            # Large positive value means perturbed lists are more similar
            perturb_effects[i] = origin_cdf_diff - perturbed_cdf_diff
        perturb_effects = sorted(perturb_effects.items(), key=lambda i: i[1], reverse=True)
        point_ranks = []
        for point_id, _ in perturb_effects:
            point_ranks.append(point_id)
        return point_ranks

    def __init__(self, test_list: List, reference_list: List, sign_level: float, ranked_list=None, k_max=100):
        """
        Croce F, Hein M. Sparse and imperceivable adversarial attacks.
        In Proceedings of the IEEE International Conference on Computer Vision 2019 (pp. 4724-4732).

        This class implement the CornerSearch algorithm in the paper
        :param ranked_list: If ranked_list is not given by users, use the paper's method to rank points
        """
        self.test_list = test_list
        self.reference_list = reference_list
        self.sign_level = sign_level
        self.N_iter = 1000
        # self.k_max = len(test_list)
        self.k_max = k_max
        # Compute point ranks and only keep the top N points
        if ranked_list is None:
            self.point_ranks = self.compute_point_importance(test_list, reference_list)[:self.k_max]
        else:
            self.point_ranks = ranked_list[:self.k_max]

        self.sample_weights = [0] * len(self.point_ranks)
        N = len(self.point_ranks)
        for rank, point_idx in enumerate(self.point_ranks):
            _rank = rank + 1 # rank starts from 1
            p = (2 * N - 2 * _rank + 1) / (N * N)
            self.sample_weights[rank] = p
        self.point_ranks = np.array(self.point_ranks)

    def generate_counterfactual(self):
        # First enumerate one-point perturbation
        for i in range(len(self.test_list)):
            _test_list = self.test_list[0:i] + self.test_list[i + 1:]
            # Return True if there is a significant difference between the two samples
            if not do_ks_test(reference_list=self.reference_list, test_list=_test_list, sign_level=self.sign_level):
                return [i, ]

        removed_points = []
        for k in tqdm.tqdm(range(2, self.k_max)):
            for i in range(self.N_iter):
                # Sample k modifications
                removed_points = set(np.random.choice(self.point_ranks, size=k, p=self.sample_weights, replace=False).tolist())
                # Remove selected points
                _test_list = [v for idx, v in enumerate(self.test_list) if idx not in removed_points]
                if len(_test_list) == 1:
                    assert do_ks_test(reference_list=self.reference_list, test_list=_test_list, sign_level=self.sign_level) == False, (_test_list)
                if not do_ks_test(reference_list=self.reference_list, test_list=_test_list, sign_level=self.sign_level):
                    return removed_points
        # Cannot generate counterfactual interpretations within given time
        return removed_points
