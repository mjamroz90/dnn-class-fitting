import os.path as op

import tensorflow as tf
from tensorflow_probability import distributions as tpd
import numpy as np

from src.models.clustering import cgs_utils
from utils import fs_utils
from utils import logger


@logger.log
class LatentSpaceSampler(object):

    def __init__(self, trace_pkl_path):
        self.trace_pkl_path = trace_pkl_path

        if isinstance(self.trace_pkl_path, str):
            assert op.exists(self.trace_pkl_path)
            self.trace_obj = fs_utils.read_pickle(trace_pkl_path)
            self.logger.info("Read trace from path: %s" % trace_pkl_path)
        else:
            self.trace_obj = trace_pkl_path

        self.logger.info("Keys from trace are: %s" % str(list(self.trace_obj.keys())))

        self.clusters_num = len(self.trace_obj['cluster_params']['mean'])
        assert self.clusters_num > 0

        self.logger.info("Clusters number in the trace: %d" % self.clusters_num)

        self.data_dim = int(self.trace_obj['cluster_params']['mean'][0].shape[0])
        self.nu_0 = cgs_utils.init_nu_0(self.trace_obj['cluster_params']['cov_chol'][0])
        assert self.data_dim > 0

        self.logger.info("Data dimensionality: %d" % self.data_dim)
        self.cluster_counts = {i: len(self.trace_obj['cluster_assignment'][i]) for i in range(self.clusters_num)}

    def sample_latent_vecs_for_cluster(self, cluster_index, samples_num):
        assert cluster_index < self.clusters_num
        t_student_distr = self.prepare_t_student_distr_obj(cluster_index)
        result = t_student_distr.sample(sample_shape=samples_num)

        return result

    def sample_latent_vecs_from_mixture(self, samples_num):
        t_student_mixture = self.prepare_t_student_mixture()
        result = t_student_mixture.sample(sample_shape=samples_num)

        return result

    @staticmethod
    def t_student_distr_from_params(params):
        dof, mean, cov_chol = params['df'], params['mean'], params['cov_chol']
        t_student_distr = tpd.MultivariateStudentTLinearOperator(df=dof, loc=mean,
                                                                 scale=tf.linalg.LinearOperatorLowerTriangular(
                                                                     cov_chol))
        return t_student_distr

    def prepare_t_student_distr_obj(self, cluster_index):
        # Given D = {x_1, x_2, ..., x_N}
        # Likelihood: P(D | mu, Sigma) = \prod_{i=1}^N  N(mu, Sigma)
        # Prior: P(mu, Sigma) = NIW(mu, Sigma | mu_0, kappa_0, nu_0, S_0)
        # Posterior: P(mu, Sigma | D) = NIW(mu, Sigma | mu_N, kappa_N, nu_N, S_N)
        # In trace object for each cluster index there are keys: mean, and cov_chol which contains quantities:
        # mu_N and chol(S_N) for posterior distribution over parameters mu, Sigma
        # Now, we want to marginalize out these parameters (i.e mu, Sigma) and get posterior predictive distribution:
        # P(x_new |  D) = \int P(x_new | mu, Sigma) * P(mu, Sigma | D) dmu dSigma,
        # this distribution has a functional form of:
        # T-Student distribution with certain parameters, these are:
        # P(x | D) = T-Student(x | mu_N, (kappa_N + 1)/(kappa_N * (nu_N - D + 1)) * S_N, nu_N - D + 1)
        t_student_params = self.prepare_t_student_params(cluster_index)
        return self.t_student_distr_from_params(t_student_params)

    def prepare_t_student_mixture(self):
        cluster_weights_unnorm = self.get_cluster_unnormalized_weights()
        cluster_weights_norm = cluster_weights_unnorm / np.sum(cluster_weights_unnorm)

        cat_distr = tpd.Categorical(probs=tf.cast(cluster_weights_norm, dtype=tf.float32))
        t_student_params = [self.prepare_t_student_params(ind) for ind in range(self.clusters_num)]
        dofs = tf.constant([t_student_params[ind]['df'] for ind in range(self.clusters_num)], dtype=tf.float32)
        means = tf.stack([t_student_params[ind]['mean'] for ind in range(self.clusters_num)])
        cov_chols = tf.stack([t_student_params[ind]['cov_chol'] for ind in range(self.clusters_num)])

        t_student_distr = tpd.MultivariateStudentTLinearOperator(df=dofs, loc=means,
                                                                 scale=tf.linalg.LinearOperatorLowerTriangular(
                                                                     cov_chols))
        t_student_mixture = tpd.MixtureSameFamily(mixture_distribution=cat_distr,
                                                  components_distribution=t_student_distr)

        return t_student_mixture

    def prepare_t_student_params(self, cluster_index):
        mean = self.trace_obj['cluster_params']['mean'][cluster_index]
        cov_chol = self.trace_obj['cluster_params']['cov_chol'][cluster_index]
        cluster_count = self.cluster_counts[cluster_index]

        kappa_0 = cgs_utils.init_kappa_0()
        kappa_n, nu_n = kappa_0 + cluster_count, self.nu_0 + cluster_count
        dof = nu_n - self.data_dim + 1

        scale_cov_factor = np.sqrt((kappa_n + 1) / (kappa_n * dof))
        scaled_cov_chol = scale_cov_factor * cov_chol

        return {'df': dof, 'mean': mean, 'cov_chol': scaled_cov_chol}

    def get_cluster_unnormalized_weights(self):
        trace_alpha = float(self.trace_obj['alpha'])
        # cluster counts is a list of length: clusters_num, where at position i is contained a size of cluster i
        cluster_counts = [x[1] for x in sorted(list(self.cluster_counts.items()), key=lambda x: x[0])]

        weights = trace_alpha + np.array(cluster_counts)
        return weights
