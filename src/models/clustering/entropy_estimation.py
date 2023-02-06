import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow_probability import distributions as tpd

from src.models.clustering import latent_space_sampler
from src.models.clustering import cgs_utils
from utils import logger
from utils import fs_utils


@logger.log
class EntropyEstimator(latent_space_sampler.LatentSpaceSampler):

    def __init__(self, trace_pkl_path, samples_num, entropy_type, **kwargs):
        super().__init__(trace_pkl_path)
        self.samples_num = samples_num

        self.nu_0 = cgs_utils.init_nu_0(self.trace_obj['cluster_params']['cov_chol'][0])
        self.kappa_0 = cgs_utils.init_kappa_0()

        assert entropy_type in {'differential', 'relative'}
        self.entropy_type = entropy_type

        if entropy_type == 'relative' and ('data_trace_path' not in kwargs and 'data' not in kwargs):
            raise ValueError("For entropy_type == 'relative' either 'data_trace_path' or 'data' "
                             "should be give in kwargs")

        self.t_student_mixture = self.prepare_t_student_mixture()
        self.kwargs = kwargs

        if entropy_type == 'relative':
            self.mvn_diag_distr = self.__get_invariant_distr()

    def estimate_entropy_with_sampling(self):
        log_probs_val = self.estimate_entropy(self.samples_num).numpy()
        self.logger.info("Computed mean of log probs for samples, val = %.3f" % log_probs_val)
        if self.entropy_type == 'differential':
            return -log_probs_val
        else:
            return log_probs_val

    def sample_points_from_mixture(self):
        samples_generated = self.t_student_mixture.sample(sample_shape=(self.samples_num,))
        self.logger.info("Finished sampling points, returning array of shape: %s" % str(samples_generated.shape))

        return samples_generated

    def sample_latent_z(self, samples_num):
        # Here we're having a posterior p(weights | z) where z is the cluster assignment
        alphas = self.get_cluster_unnormalized_weights()
        sampled_weights = stats.dirichlet.rvs(alpha=alphas, size=samples_num)

        self.logger.info("Sampled latent weights of shape: %s" % str(sampled_weights.shape))
        # Here we're sampling categorical variables z given the previously sampled weights
        sampled_z = tpd.Categorical(probs=sampled_weights).sample().numpy()
        self.logger.info("Sampled latent z of shape: %s" % str(sampled_z.shape))

        return sampled_z

    @tf.function
    def estimate_entropy(self, samples_num):
        sampled_points = tf.cast(self.t_student_mixture.sample(sample_shape=(samples_num,)), tf.float32)
        log_probs = self.t_student_mixture.log_prob(sampled_points)

        if self.entropy_type == 'relative':
            log_probs -= self.mvn_diag_distr.log_prob(sampled_points)

        return tf.reduce_mean(log_probs)

    def __get_invariant_distr(self):
        if 'data_trace_path' in self.kwargs:
            data_trace = fs_utils.read_pickle(self.kwargs['data_trace_path'])
            data_arr = data_trace['data'].astype(np.float32)
        else:
            data_arr = self.kwargs['data']

        self.logger.info("Creating diagonal MVN distribution for computing relative entropy, fetched original "
                         "data of shape: %s" % str(data_arr.shape))
        scale_diagonal = np.sqrt(np.diag(cgs_utils.init_cov(data_arr)))
        data_mean = np.mean(data_arr, axis=0)

        mvn_diag_distr = tpd.MultivariateNormalDiag(loc=tf.constant(data_mean, dtype=tf.float32),
                                                    scale_diag=tf.constant(scale_diagonal, dtype=tf.float32))
        return mvn_diag_distr

