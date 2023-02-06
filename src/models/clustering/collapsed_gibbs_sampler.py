import os.path as op
import pickle
import random
from datetime import datetime

from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tpd

from utils import prob_utils
from utils import fs_utils
from utils.logger import log


@log
class CollapsedGibbsSampler(object):

    def __init__(self, init_strategy, max_clusters_num, batch_size=1, out_dir=None, **kwargs):
        self.init_strategy = init_strategy
        self.max_clusters_num = max_clusters_num

        self.a = kwargs['a'] if 'a' in kwargs else 1.0
        self.b = kwargs['b'] if 'b' in kwargs else 1.0

        self.batch_size = batch_size

        self.alpha = self.__sample_alpha()
        self.logger.info("Sampled initial alpha, it's value: %.3f" % self.alpha)
        self.logger.info("Batch size is set to %d" % self.batch_size)

        self.out_dir = out_dir

        self.skip_epochs_logging = kwargs['skip_epochs_logging'] if 'skip_epochs_logging' in kwargs else 1
        self.skip_epochs_ll_calc = kwargs['skip_epochs_ll_calc'] if 'skip_epochs_ll_calc' in kwargs else 1
        self.restore_snapshot_pkl_path = kwargs['restore_snapshot_pkl_path'] if 'restore_snapshot_pkl_path' in kwargs \
            else None

    def fit(self, iterations_num, data):
        from src.models.clustering import config
        config.DATA_DIM = data.shape[1]
        from src.models.clustering import cgs_utils

        if self.restore_snapshot_pkl_path is not None:
            snapshot = fs_utils.read_pickle(self.restore_snapshot_pkl_path)
            curr_alpha, ass_ll = snapshot['alpha'], snapshot['ass_ll']
            cluster_assignment = snapshot['cluster_assignment']
            examples_assignment = self.get_examples_assignment(cluster_assignment)
            n_comps = len(cluster_assignment)
            cluster_params = snapshot['cluster_params']
            self.logger.info("Restored params from snapshot path: %s, clusters num: %d" %
                             (self.restore_snapshot_pkl_path, n_comps))
            init_iter = int(op.splitext(op.basename(self.restore_snapshot_pkl_path))[0].split('_')[-1]) + 1
        else:
            cluster_assignment, examples_assignment = self.get_initial_assignment(data)
            curr_alpha, n_comps = self.alpha, len(cluster_assignment)
            cluster_params = self.assign_initial_params(cluster_assignment, data, n_comps)
            init_iter = 0
            self.logger.info("Initialized params for first assignment")
            self.logger.info("Chosen first assignment, clusters num: %d" % n_comps)

        n_points = data.shape[0]

        mean_0, cov_0 = self.initial_mean0_cov0(data, data, n_comps)
        nu_0 = cgs_utils.init_nu_0(data)
        cov_chol_0 = np.linalg.cholesky(cov_0)

        init_values = {'means': np.array(cluster_params['mean']), 'cov_chols': np.array(cluster_params['cov_chol']),
                       'mean_0': mean_0, 'cov_chol_0': cov_chol_0}

        tf_computation_manager = cgs_utils.TfCgsSharedComputationsManager(data, len(cluster_assignment), init_values,
                                                                          self.batch_size)
        self.logger.info("Instantiated computations manager")
        ex_permutation = list(range(data.shape[0]))

        for iter_num in range(init_iter, iterations_num, 1):
            if iter_num % self.skip_epochs_ll_calc == 0:
                ass_ll = self.data_log_likelihood(cluster_assignment, data, cluster_params, nu_0)
                self.logger.info("Started %d epoch, current  clusters number: %d, assignment ll: %.2f, "
                                 "curr-alpha: %.2f " % (iter_num, len(cluster_assignment), ass_ll, curr_alpha))

            last_log_counter = 0

            for start_batch_index in range(0, data.shape[0], self.batch_size):
                curr_alpha = self.update_alpha(curr_alpha, n_points, len(cluster_assignment))
                data_point_indices = ex_permutation[start_batch_index: start_batch_index + self.batch_size]
                cluster_counts_before_removal = [len(cluster_examples) for cluster_examples in cluster_assignment]
                data_batch_clusters, clusters_batch_removed = self.remove_assignment_for_batch(data_point_indices,
                                                                                               cluster_assignment,
                                                                                               examples_assignment)
                # At last data point, fetch current means and covariances for clusters
                return_curr_params = True if data_point_indices[-1] == ex_permutation[-1] else False
                curr_cluster_counts = [len(cluster_examples) for cluster_examples in cluster_assignment]

                res = tf_computation_manager.sample_clusters_for_data_batch(data_point_indices, data_batch_clusters,
                                                                            clusters_batch_removed,
                                                                            cluster_counts_before_removal,
                                                                            curr_cluster_counts, curr_alpha,
                                                                            return_curr_params=return_curr_params)
                z_indices = res['sampled_clusters']
                new_cluster_index = len(cluster_assignment)
                # Very rare error that tf sampling returns index beyond the domain, because of numerical issues
                z_indices = [new_cluster_index if z_indx > new_cluster_index else z_indx for z_indx in z_indices]
                for z_indx, data_point_indx in zip(z_indices, data_point_indices):
                    if z_indx == new_cluster_index:
                        if len(cluster_assignment) == new_cluster_index:
                            cluster_assignment.append({data_point_indx})
                        else:
                            # len(cluster_assignment) > new_cluster_index
                            cluster_assignment[z_indx].add(data_point_indx)
                    else:
                        cluster_assignment[z_indx].add(data_point_indx)

                    examples_assignment[data_point_indx] = z_indx

                if int(start_batch_index / 100) > last_log_counter:
                    last_log_counter = int(start_batch_index / 100)
                    timestamp_str = datetime.now().strftime("%H:%M:%S")
                    self.logger.info("Iteration: %d, example: %d/%d, current clusters num: %d, %s, alpha: %.2f" %
                                     (iter_num, start_batch_index + len(data_point_indices), data.shape[0],
                                      len(cluster_assignment), timestamp_str, curr_alpha))

            cluster_params = {'mean': res['mean'], 'cov_chol': res['cov_chol']}
            if self.out_dir is not None and iter_num % self.skip_epochs_logging == 0:
                self.save_model(cluster_params, cluster_assignment, data, iter_num, curr_alpha, ass_ll)
                self.logger.info("Saved model from iteration: %d" % iter_num)

            random.shuffle(ex_permutation)

        return cluster_params

    def remove_assignment_for_batch(self, data_point_indices, cluster_assignment, examples_assignment):
        # In clusters_removed we wanna store info if cluster was removed
        batch_clusters, clusters_removed = [], {}
        for i, data_point_index in enumerate(data_point_indices):
            data_point_cluster, cluster_removed_after_data_point = self.get_data_point_cluster(
                data_point_index, cluster_assignment, examples_assignment)
            batch_clusters.append(data_point_cluster)
            clusters_removed[data_point_cluster] = cluster_removed_after_data_point

        clusters_to_remove = set([cluster for cluster, cluster_to_remove in clusters_removed.items()
                                  if cluster_to_remove])
        self.remove_clusters(clusters_to_remove, cluster_assignment, examples_assignment)
        return batch_clusters, clusters_removed

    def get_data_point_cluster(self, data_point_indx, cluster_assignment, examples_assignment):
        data_point_cluster = examples_assignment[data_point_indx]
        cluster_assignment[data_point_cluster].remove(data_point_indx)

        # not necessary, but for the sake of clarification
        examples_assignment[data_point_indx] = -1

        if len(cluster_assignment[data_point_cluster]) == 0:
            self.logger.info("Cluster %d will be removed" % data_point_cluster)
            return data_point_cluster, True
        else:
            return data_point_cluster, False

    @staticmethod
    def remove_clusters(clusters_to_remove, cluster_assignment, examples_assignment):
        new_cluster_assignment = [ass for index, ass in enumerate(cluster_assignment)
                                  if index not in clusters_to_remove]
        cluster_assignment.clear()

        for cluster_idx, cluster_examples in enumerate(new_cluster_assignment):
            cluster_assignment.append(cluster_examples)
            for ex in cluster_examples:
                examples_assignment[ex] = cluster_idx

    @tf.function
    def log_likelihood(self, data, mean, cov_chol):
        dist = tpd.MultivariateNormalTriL(loc=mean, scale_tril=cov_chol)
        return dist.log_prob(data)

    def data_log_likelihood(self, cluster_assignment, data, cluster_params, nu_0):
        # sampling covariance from marginal p(sigma | D) for each cluster k=1,2,...,K
        # sampling mean from marginal p(mean | D) for each cluster k=1,2,...,K
        # sampling pi_k for each cluster k=1,2,...,K
        # p(sigma | D) = IW(S_n, nu_n)
        # p(mean | D) = t_student_pdf(mean | m_n, [1/(kappa_n)(nu_n-D+1)]S_n, nu_n-D+1)
        # p(pi | z) = Dir({alpha_k + \sum_{i=1}^N I(z_i=k) })
        clusters_num = len(cluster_assignment)
        examples_assignment = [0] * data.shape[0]
        for cluster, cluster_examples in enumerate(cluster_assignment):
            for ex in cluster_examples:
                examples_assignment[ex] = cluster

        data_dim = data.shape[1]
        means, cov_chols = self.__sample_marginals_for_mean_and_sigma(cluster_assignment, cluster_params, nu_0,
                                                                      data_dim)
        data_log_pdfs = []
        for cov_chol, mean in zip(cov_chols, means):
            mean, cov_chol = mean.astype(data.dtype), cov_chol.astype(data.dtype)
            k_log_pdfs = self.log_likelihood(data, mean, cov_chol)
            data_log_pdfs.append(k_log_pdfs)

        data_log_pdfs = np.array(data_log_pdfs, dtype=data.dtype)
        assignment_ll = np.sum(data_log_pdfs[examples_assignment, np.arange(data.shape[0])])
        return float(assignment_ll)

    @staticmethod
    def __sample_marginals_for_mean_and_sigma(cluster_assignment, cluster_params, nu_0, data_dim):
        from src.models.clustering import cgs_utils

        sigmas_chols, means = [], []

        for k, k_examples in enumerate(cluster_assignment):
            k_mean, k_cov_chol = cluster_params['mean'][k], cluster_params['cov_chol'][k]
            nu_k, kappa_k = nu_0 + len(k_examples), cgs_utils.init_kappa_0() + len(k_examples)

            s_k = np.dot(k_cov_chol, k_cov_chol.T)
            sigma_k = stats.invwishart.rvs(df=nu_k, scale=s_k)
            mean_k = prob_utils.multivariate_t_rvs(k_mean, np.sqrt(1. / (kappa_k * (nu_k - data_dim + 1))) * k_cov_chol,
                                                   df=nu_k - data_dim + 1)

            sigmas_chols.append(np.linalg.cholesky(sigma_k).astype(np.float32))
            means.append(mean_k.astype(np.float32))

        return means, sigmas_chols

    @staticmethod
    def __sample_weights(cluster_assignment, k, alpha):
        alpha_k = np.ones(k) * (alpha / k)
        alpha_k += [len(c) for c in cluster_assignment]
        return stats.dirichlet.rvs(alpha=alpha_k, size=1)[0]

    def get_initial_assignment(self, data):
        clusters_num = min(data.shape[0], self.max_clusters_num)
        init = np.random.randint(0, clusters_num, size=data.shape[0])
        clusters_examples = {}
        for example_idx, example_cluster in enumerate(init):
            if example_cluster in clusters_examples:
                clusters_examples[example_cluster].add(example_idx)
            else:
                clusters_examples[example_cluster] = {example_idx}

        cluster_assignment = [examples for _, examples in clusters_examples.items()]
        examples_assignment = self.get_examples_assignment(cluster_assignment)

        return cluster_assignment, examples_assignment

    @staticmethod
    def get_examples_assignment(cluster_assignment):
        from itertools import chain

        n_points = len(list(chain.from_iterable(cluster_assignment)))
        examples_assignment = [0] * n_points

        for cluster_idx, examples in enumerate(cluster_assignment):
            for ex in examples:
                examples_assignment[ex] = cluster_idx

        return examples_assignment

    def assign_initial_params(self, assignment, data, n_comps):
        cluster_params = {'mean': [], 'cov_chol': []}

        for cluster_num, examples in enumerate(assignment):
            cluster_data = data[list(examples), :]

            cluster_mean, cluster_cov_chol = self.initialize_params_for_samples(data, cluster_data, n_comps)
            cluster_params['mean'].append(cluster_mean.astype(data.dtype))
            cluster_params['cov_chol'].append(cluster_cov_chol.astype(data.dtype))

        return cluster_params

    def initialize_params_for_samples(self, data, cluster_data, n_comps):
        from src.models.clustering import cgs_utils

        kappa_0 = cgs_utils.init_kappa_0()
        n_cluster = cluster_data.shape[0]
        sample_data_mean = np.mean(cluster_data, axis=0)

        cluster_mean_0, cluster_cov_0 = self.initial_mean0_cov0(data, cluster_data, n_comps)

        cluster_mean = (kappa_0 * cluster_mean_0 + n_cluster * sample_data_mean) / (kappa_0 + n_cluster)

        cluster_cov = cluster_cov_0 + np.dot(cluster_data.T, cluster_data)
        cluster_cov += kappa_0 * np.outer(cluster_mean_0, cluster_mean_0)
        cluster_cov -= (kappa_0 + n_cluster) * np.outer(cluster_mean, cluster_mean)
        return cluster_mean, np.linalg.cholesky(cluster_cov)

    def initial_mean0_cov0(self, data, cluster_data, n_comps):
        if self.init_strategy == 'init_per_init_cluster':
            cluster_mean_0 = self.init_mean(cluster_data)
            if cluster_data.shape[0] > 1:
                cluster_cov_0 = self.init_cov(cluster_data, n_comps)
            else:
                cluster_cov_0 = self.init_cov_eye(cluster_data)
        elif self.init_strategy == 'init_randomly':
            cluster_mean_0 = self.init_mean_random(cluster_data)
            cluster_cov_0 = self.init_cov_random(cluster_data)
        elif self.init_strategy == 'init_eye':
            cluster_mean_0 = self.init_mean(cluster_data)
            cluster_cov_0 = self.init_cov_eye(cluster_data)
        elif self.init_strategy == 'init_data_stats':
            cluster_mean_0, cluster_cov_0 = self.init_mean(data), self.init_cov(data, n_comps)
        else:
            raise ValueError("Unknown initialization: %s" % self.init_strategy)
        return cluster_mean_0, cluster_cov_0

    def __sample_alpha(self):
        return stats.gamma.rvs(self.a, self.b)

    def update_alpha(self, old_alpha, n_points, k):
        u = stats.bernoulli.rvs(float(n_points) / (n_points + old_alpha))
        v = stats.beta.rvs(old_alpha + 1., n_points)

        new_alpha = np.random.gamma(self.a + k - 1 + u, 1. / (self.b - np.log(v)))
        return new_alpha

    @staticmethod
    def init_cov(data, num_components):
        data_mean = np.mean(data, axis=0)
        data_norm = data - np.expand_dims(data_mean, axis=0)

        data_var = np.dot(data_norm.T, data_norm) * (1. / data.shape[0])
        div_factor = np.power(num_components, 2. / data.shape[1])

        return np.diag(np.diag(data_var / div_factor))

    @staticmethod
    def init_cov_eye(data):
        return np.eye(data.shape[1], dtype=np.float64)

    @staticmethod
    def init_cov_random(data):
        d = data.shape[1]
        vec = np.random.randn(d, d)
        cov = np.dot(vec.T, vec)
        return cov

    @staticmethod
    def init_mean(data):
        return np.mean(data, axis=0)

    @staticmethod
    def init_mean_random(data):
        d = data.shape[1]
        return np.random.randn(d)

    def save_model(self, cluster_params, cluster_assignment, data, it_index, curr_alpha, ll):
        import pickle

        obj = {'cluster_assignment': cluster_assignment, 'cluster_params': cluster_params,
               'init_strategy': self.init_strategy, 'alpha': curr_alpha, 'ass_ll': ll}
        try:
            if it_index == 0:
                obj.update({'data': data})
            with open(op.join(self.out_dir, "cgs_%d.pkl" % it_index), 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            self.logger.error(e)
