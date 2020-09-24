from typing import List
import numpy as np
import tqdm
from kstest.ks_test import do_ks_test


class Grace:
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
        max_cdf_diff = max([abs(reference_set_cdf[v] - test_set_cdf[v]) for v in all_values])
        m = len(test_list)
        n = len(reference_list)
        weight_cdf_diff = max_cdf_diff * np.sqrt((n * m) / (n + m))
        return weight_cdf_diff

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
        Thai Le, Suhang Wang, and Dongwon Lee. 2020.
        GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model's Prediction.
        In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20).

        This class implement the Grace algorithm in the paper

        NOTE:
        The entropy based feature selection method and the local points based feature selection method are not implemented
        as those methods requires training data.

        :param ranked_list: The rank list is provide by users.
        """
        self.reference_list = reference_list
        self.sign_level = sign_level
        self.step = 100
        # self.k_max = len(test_list)
        self.k_max = k_max
        self.test_list = np.array(test_list)
        self.critical_value = np.sqrt(- np.log(sign_level / 2) * 0.5)
        # Compute point ranks and only keep the top N points
        if ranked_list is None:
            self.point_ranks = self.compute_point_importance(test_list, reference_list)[:self.k_max]
        else:
            self.point_ranks = ranked_list[:self.k_max]

    def compute_gradient(self, one_hot_adv_test_list):
        """
        Cheng, Minhao, et al.
        "Query-efficient hard-label black-box attack: An optimization-based approach."
        arXiv preprint arXiv:1807.04457 (2018).

        Implement the gradient estimation method used in the paper.
        """
        q = 20  # The parameter is adopted from the paper
        beta = 0.5  # The parameter is adopted from the paper
        _one_hot_adv_test_list = np.clip(one_hot_adv_test_list, 0, 1)
        _one_hot_adv_test_list = np.rint(_one_hot_adv_test_list).astype(np.float64)
        test_list = self.test_list[_one_hot_adv_test_list == 1].tolist()
        weight_cdf_diff = self.compute_max_cdf(test_list, self.reference_list)
        est_gradient = np.array([0.0] * len(one_hot_adv_test_list))
        i = 0
        while i < q:
            # Random sample a perturbation from normal distribution
            u = np.random.normal(0, 1, size=len(one_hot_adv_test_list))
            _theta_prime = one_hot_adv_test_list + beta * u
            _theta_prime = np.clip(_theta_prime, 0, 1)
            _theta_prime_round = np.rint(_theta_prime).astype(np.float64)
            _test_list = self.test_list[_theta_prime_round == 1].tolist()
            if len(_test_list) > 0:
                _weight_cdf_diff = self.compute_max_cdf(_test_list, self.reference_list)
                # The estimated gradient
                gradient = (_weight_cdf_diff - weight_cdf_diff) / beta * (u + 1e-8)
                est_gradient += gradient
                # clip the gradient to avoid infinity value
                est_gradient = np.clip(est_gradient, -1e5, 1e5)
                i += 1
        # Smooth the estimated gradient by taking average over q estimated gradients
        return est_gradient / q, weight_cdf_diff

    def generate_counterfactual(self):
        one_hot_adv_test_list = np.array([1.0] * len(self.test_list))
        for k in tqdm.tqdm(range(1, self.k_max)):
            top_k_features = set(self.point_ranks[:k])
            gradient_mask = np.array([1.0 if i in top_k_features else 0.0 for i in range(len(self.test_list))])
            for i in range(self.step):
                gradient, weight_cdf_diff = self.compute_gradient(one_hot_adv_test_list)
                nornimator = weight_cdf_diff - self.critical_value
                # 1e-5 is added to avoid 0 denorm
                denorm = np.linalg.norm(gradient, 2) + 1e-5
                step_size = nornimator / (denorm * denorm)
                updt = step_size * gradient
                # Only update the top-K selected features
                updt = updt * gradient_mask
                # Apply gradient descend to minimize the CDF difference
                one_hot_adv_test_list = one_hot_adv_test_list - (1 + 0.02) * updt
                # clip the gradient to avoid infinity value
                one_hot_adv_test_list = np.clip(one_hot_adv_test_list, -1e5, 1e5)
                # Evaluate the KS test
                _one_hot_adv_test_list = np.clip(one_hot_adv_test_list, 0, 1)
                _one_hot_adv_test_list = np.rint(_one_hot_adv_test_list).astype(np.float64)
                test_list = self.test_list[_one_hot_adv_test_list == 1].tolist()
                # print(len(test_list))
                if len(test_list) == 0:
                    interpretation = {i for i in range(len(self.test_list)) if _one_hot_adv_test_list[i] == 0}
                    return interpretation
                if not do_ks_test(reference_list=self.reference_list, test_list=test_list, sign_level=self.sign_level):
                    interpretation = {i for i in range(len(self.test_list)) if _one_hot_adv_test_list[i] == 0}
                    return interpretation
                # one_hot_adv_test_list = _one_hot_adv_test_list
        # Cannot generate counterfactual interpretation within given time
        one_hot_adv_test_list = np.clip(one_hot_adv_test_list, 0, 1)
        one_hot_adv_test_list = np.rint(one_hot_adv_test_list).astype(np.float64)
        interpretation = {i for i in range(len(self.test_list)) if one_hot_adv_test_list[i] == 0}
        return interpretation


def test():
    from kstest.mochi import Interpreter
    from kstest.utils import set_random_seed
    set_random_seed()
    test_list = np.random.normal(0, 1, 100).tolist()
    reference_list = np.random.normal(1, 1, 100).tolist()
    rank_list = list(range(len(test_list) - 1, -1, -1))
    interpret = Interpreter(test_list, reference_list, rank_list, 0.05, True, True)
    k = interpret.find_k()
    grace = Grace(test_list, reference_list, 0.05, rank_list)
    interpretation = grace.generate_counterfactual()
    print(interpretation)
    print("Ground Truth K", k)
    print("Grace K", len(interpretation))


if __name__ == '__main__':
    test()
