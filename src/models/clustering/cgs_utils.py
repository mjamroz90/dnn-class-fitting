from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python import math
import numpy as np

from src.models.clustering.lib import ops
from utils.logger import log
from src.models.clustering import config


@tf.function(input_signature=(tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, config.DATA_DIM, config.DATA_DIM], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None], dtype=tf.float32)),
             experimental_relax_shapes=True)
def t_student_log_pdf_tf(mean_matrix, chol_cov_matrices, data_batch, nus, cluster_counts):
    data_dim = tf.constant(data_batch.get_shape()[-1], dtype=tf.float32)
    clusters_num = tf.shape(mean_matrix)[0]

    kappas = tf.constant(init_kappa_0()) + cluster_counts

    scale_fact = tf.expand_dims(tf.expand_dims((kappas + 1.) / (kappas * nus), axis=-1), axis=-1)
    chol_cov_scaled = tf.sqrt(scale_fact) * chol_cov_matrices

    chol_cov_diagonals = tf.linalg.diag_part(chol_cov_scaled)
    log_dets_sqrt = tf.reduce_sum(tf.math.log(chol_cov_diagonals), axis=-1)

    data_batch_tiled = tf.tile(tf.expand_dims(data_batch, axis=1), (1, clusters_num, 1))
    # data_batch_norm.shape -> (batch_size, clusters_num, data_dim)
    data_batch_norm = data_batch_tiled - mean_matrix
    #  data_batch_norm_transposed.shape -> (clusters_num, data_dim, batch_size)
    data_batch_norm_transposed = tf.transpose(data_batch_norm, (1, 2, 0))
    # chol_cov_scaled.shape -> (clusters_num, data_dim, data_dim)
    # vecs.shape -> (clusters_num, data_dim, batch_size)
    vecs = tf.linalg.triangular_solve(chol_cov_scaled, data_batch_norm_transposed, lower=True)
    # vecs_norm.shape -> (clusters_num, batch_size)
    vecs_norm = tf.norm(vecs, axis=1)
    # vecs_norm.shape -> (batch_size, clusters_num)
    vecs_norm = tf.transpose(vecs_norm, (1, 0))

    num = tf.math.lgamma((nus + data_dim) / 2.)

    denom = tf.math.lgamma(nus / 2.) + (data_dim / 2.) * (tf.math.log(nus) + np.log(np.pi))
    denom += log_dets_sqrt
    denom += ((nus + data_dim) / 2.) * math.log1psquare(vecs_norm / tf.sqrt(nus))

    return num - denom


def init_kappa_0():
    return 0.01


def init_nu_0(data):
    return data.shape[1] + 2


def init_cov(data):
    data_mean = np.mean(data, axis=0)
    data_norm = data - np.expand_dims(data_mean, axis=0)

    data_var = np.dot(data_norm.T, data_norm) * (1. / data.shape[0])
    return np.diag(np.diag(data_var))


@log
class TfCgsSharedComputationsManager(object):

    def __init__(self, data, init_clusters_num, init_values, batch_size_max_value):
        self.data = data
        self.data_dim = data.shape[1]
        self.init_clusters_num = init_clusters_num
        self.new_clusters_added = 0
        self.nu0 = tf.constant(init_nu_0(self.data), dtype=tf.float32)
        self.kappa0 = tf.constant(init_kappa_0(), dtype=tf.float32)
        self.batch_size_max_value = batch_size_max_value

        self.data_sv = tf.constant(data, dtype=tf.float32)
        self.cov_chols = tf.Variable(name='cov_chols', initial_value=init_values['cov_chols'].astype(np.float32),
                                     dtype=tf.float32,
                                     shape=tf.TensorShape([None, self.data_dim, self.data_dim]), trainable=False)

        self.means = tf.Variable(name='means', initial_value=init_values['means'].astype(np.float32),
                                 dtype=tf.float32, shape=tf.TensorShape([None, self.data_dim]),
                                 trainable=False)

        self.mean_0 = tf.constant(init_values['mean_0'].astype(np.float32), dtype=tf.float32)
        self.cov_chol_0 = tf.constant(init_values['cov_chol_0'].astype(np.float32), dtype=tf.float32)

        self.active_clusters = [True] * self.init_clusters_num
        self.new_old_cluster_mapping = list(range(init_clusters_num))

        self.logger.info("Assigned initial values to covariances, means and data points")

    def sample_clusters_for_data_batch(self, data_points_indices, curr_batch_clusters, clusters_batch_removed,
                                       cluster_counts_before_removal, cluster_counts_after_removal, alpha,
                                       return_curr_params=False):

        old_data_batch_clusters = [self.new_old_cluster_mapping[point_cluster] for point_cluster in curr_batch_clusters]
        # cluster mask needs to be updated after taking the info of clusters removal
        # data_batch_clusters_ns - under index i keeps an info about examples count in cluster
        # occupied by example - data_points_indices[i]
        data_batch_clusters_ns, curr_clusters_to_remove = [], set()
        for curr_data_point_cluster, curr_data_point_old_cluster in zip(curr_batch_clusters, old_data_batch_clusters):
            cluster_removed = clusters_batch_removed[curr_data_point_cluster]
            if cluster_removed:
                self.active_clusters[curr_data_point_old_cluster] = False
                curr_clusters_to_remove.add(curr_data_point_cluster)
                cluster_n = 0
                self.logger.info("Updated info about cluster to be removed: %d" % curr_data_point_cluster)
            else:
                cluster_n = cluster_counts_before_removal[curr_data_point_cluster]

            data_batch_clusters_ns.append(cluster_n)

        self.new_old_cluster_mapping = [e for i, e in enumerate(self.new_old_cluster_mapping)
                                        if i not in curr_clusters_to_remove]

        cluster_counts_t = tf.convert_to_tensor([float(c) for c in cluster_counts_after_removal], dtype=np.float32)
        data_batch_clusters_ns_t = tf.convert_to_tensor([float(c) for c in data_batch_clusters_ns], dtype=tf.float32)
        data_points_indices_t = tf.convert_to_tensor(data_points_indices, dtype=tf.int32)
        old_data_batch_clusters_t = tf.convert_to_tensor(old_data_batch_clusters, dtype=tf.int32)
        active_clusters_t = tf.convert_to_tensor(self.active_clusters, dtype=tf.bool)
        alpha_t = tf.convert_to_tensor(alpha, dtype=tf.float32)
        new_old_cluster_mapping_t = tf.convert_to_tensor(self.new_old_cluster_mapping, dtype=tf.int32)

        sampled_clusters, new_m, new_c = self.tf_sample_clusters_for_data_batch(data_points_indices_t,
                                                                                data_batch_clusters_ns_t,
                                                                                old_data_batch_clusters_t,
                                                                                cluster_counts_t,
                                                                                active_clusters_t, alpha_t,
                                                                                new_old_cluster_mapping_t)

        sampled_clusters = list(sampled_clusters.numpy())
        # Update mapping, if new cluster will be added
        if any(c == len(cluster_counts_after_removal) for c in sampled_clusters):
            self.new_clusters_added += 1
            self.new_old_cluster_mapping.append(self.init_clusters_num + self.new_clusters_added - 1)
            self.active_clusters.append(True)
            self.logger.info("Added new cluster: %d" % max(sampled_clusters))

        res = {'sampled_clusters': sampled_clusters}
        if return_curr_params:
            new_m, new_c = new_m.numpy(), new_c.numpy()
            res.update({'mean': new_m[self.active_clusters, :], 'cov_chol': new_c[self.active_clusters, :, :]})

        return res

    @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.bool),
                                  tf.TensorSpec(shape=(), dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)))
    def tf_sample_clusters_for_data_batch(self, data_point_indices, batch_clusters_ns, old_data_batch_clusters,
                                          cluster_counts, cluster_mask, alpha, new_old_cluster_mapping):

        n_points = tf.constant(self.data.shape[0], dtype=tf.float32)
        #  data_point = self.data_sv[data_point_index, :]
        data_points = tf.gather(self.data_sv, data_point_indices)
        # downdating old cluster params
        down_means, down_cov_chols = self.downdate_cluster_params_for_batch_in_aggregate(data_points,
                                                                                         old_data_batch_clusters,
                                                                                         batch_clusters_ns)

        self.means.assign(down_means)
        self.cov_chols.assign(down_cov_chols)

        nus = self.nu0 - tf.constant(self.data_dim - 1, dtype=tf.float32) + cluster_counts

        active_means = tf.boolean_mask(self.means, cluster_mask)
        active_cov_chols = tf.boolean_mask(self.cov_chols, cluster_mask)

        cluster_probs = cluster_counts / (n_points + alpha - 1.)
        # pred_post_log_probs.shape -> (batch_size, clusters_num)
        pred_post_log_probs = t_student_log_pdf_tf(active_means, active_cov_chols, data_points,
                                                   nus, cluster_counts)
        # cluster_log_preds_unnorm.shape -> (batch_size, clusters_num)
        cluster_log_preds_unnorm = tf.expand_dims(tf.math.log(cluster_probs), axis=0) + pred_post_log_probs

        new_cluster_prob = alpha / (n_points + alpha - 1.)
        # post_log_pred_new_cluster.shape -> (batch_size, 1)
        post_log_pred_new_cluster = t_student_log_pdf_tf(tf.expand_dims(self.mean_0, axis=0),
                                                         tf.expand_dims(self.cov_chol_0, axis=0),
                                                         data_points, tf.convert_to_tensor([3.], dtype=tf.float32),
                                                         tf.convert_to_tensor([0.], dtype=tf.float32))
        # new_cluster_log_prob_unnorm.shape -> (batch_size, 1)
        new_cluster_log_prob_unnorm = tf.math.log(new_cluster_prob) + post_log_pred_new_cluster

        # all_logs_probs_unnormalized.shape -> (batch_size, clusters_num + 1)
        all_logs_probs_unnormalized = tf.concat([cluster_log_preds_unnorm, new_cluster_log_prob_unnorm], axis=1)
        # tf.print("Unnormalized log probs: ", tf.reduce_any(tf.math.is_nan(all_logs_probs_unnormalized)))

        # norm_const.shape -> (batch_size, 1)
        norm_const = tf.reduce_logsumexp(all_logs_probs_unnormalized, axis=-1, keepdims=True)
        all_probs_log_normalized = all_logs_probs_unnormalized - norm_const
        sampled_clusters = tf.cast(tf.reshape(tf.random.categorical(logits=all_probs_log_normalized, num_samples=1),
                                              [-1]), tf.int32)

        upd_means, upd_cov_chols = self.update_cluster_params_for_batch_in_aggregate(sampled_clusters, data_points,
                                                                                     cluster_counts,
                                                                                     new_old_cluster_mapping)

        self.means.assign(upd_means)
        self.cov_chols.assign(upd_cov_chols)

        return sampled_clusters, upd_means, upd_cov_chols

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.float32)))
    def downdate_cluster_params_for_batch_in_aggregate(self, data_batch, old_data_batch_clusters,
                                                       data_batch_clusters_ns):
        def add_indices(curr_clusters_data_points, i, curr_clusters_counts, curr_max):
            cluster = non_empty_clusters_set[i]
            cluster_data_points_indices = tf.cast(tf.reshape(tf.where(old_data_batch_clusters == cluster), [-1]),
                                                  tf.int32)
            cluster_size = tf.cast(tf.shape(cluster_data_points_indices)[0], tf.int32)
            empty_ = tf.zeros(shape=(self.batch_size_max_value - cluster_size,), dtype=tf.int32) - 1
            # We are filling -1 to cluster data point indices, to have batch_size at the end
            new_ta = curr_clusters_data_points.write(i, tf.concat([cluster_data_points_indices, empty_], axis=0))
            cluster_count = tf.cast(data_batch_clusters_ns[cluster_data_points_indices[0]], tf.float32)
            new_ta1 = curr_clusters_counts.write(i, cluster_count)
            new_max = tf.cond(tf.greater(cluster_size, curr_max), true_fn=lambda: cluster_size,
                              false_fn=lambda: curr_max)
            return new_ta, i + 1, new_ta1, new_max

        def downdate_params_in_aggregate(i, means_, cov_chols_, counts_):
            i_data_points_indices = clusters_data_points_mtx[:, i]
            i_non_empty_indices = tf.reshape(tf.where(i_data_points_indices != -1), [-1])
            i_data_points_indices_valid = tf.gather(i_data_points_indices, i_non_empty_indices)
            i_data_points = tf.gather(data_batch, i_data_points_indices_valid)
            i_means, i_cov_chols = tf.gather(means_, i_non_empty_indices), tf.gather(cov_chols_, i_non_empty_indices)
            i_counts = tf.gather(counts_, i_non_empty_indices)
            new_i_means, new_i_cov_chols = downdate_(i_data_points, i_means, i_cov_chols, i_counts)
            new_means_ = tf.tensor_scatter_nd_update(means_, tf.expand_dims(i_non_empty_indices, axis=1), new_i_means)
            new_cov_chols_ = tf.tensor_scatter_nd_update(cov_chols_, tf.expand_dims(i_non_empty_indices, axis=1),
                                                         new_i_cov_chols)

            return i + 1, new_means_, new_cov_chols_, counts_ - 1.

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, config.DATA_DIM, config.DATA_DIM], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None], dtype=tf.float32)))
        def downdate_(data_points, means_, cov_chols_, counts_):
            c = tf.expand_dims(self.kappa0 + counts_, axis=1)
            new_means = (means_ * c - data_points) / (c - 1.)
            multipliers = tf.reshape(-c / (c - 1.), [-1])

            new_cov_chols = ops.cholesky_update(cov_chols_, data_points - means_, multiplier=multipliers)
            return new_means, new_cov_chols

        points_indices_with_non_empty_clusters = tf.reshape(tf.where(data_batch_clusters_ns != 0.), [-1])
        points_non_empty_clusters = tf.gather(old_data_batch_clusters, points_indices_with_non_empty_clusters)
        non_empty_clusters_set, _ = tf.unique(points_non_empty_clusters)

        clusters_num = tf.shape(non_empty_clusters_set)[0]
        clusters_data_points = tf.TensorArray(dtype=tf.int32, size=clusters_num,
                                              element_shape=(self.batch_size_max_value,))
        clusters_counts = tf.TensorArray(dtype=tf.float32, size=clusters_num, element_shape=())
        state = (clusters_data_points, tf.constant(0), clusters_counts, tf.constant(-1, dtype=tf.int32))
        filled_clusters_data_points, _, filled_clusters_counts, max_cluster_size = tf.while_loop(
            lambda t1, i, t2, m: tf.less(i, clusters_num), add_indices, loop_vars=state)

        clusters_data_points_mtx = filled_clusters_data_points.stack()
        clusters_counts_mtx = filled_clusters_counts.stack()

        init_means = tf.gather(self.means, non_empty_clusters_set)
        init_cov_chols = tf.gather(self.cov_chols, non_empty_clusters_set)

        state = (tf.constant(0), init_means, init_cov_chols, clusters_counts_mtx)
        _, final_means, final_cov_chols, _ = tf.while_loop(lambda i, m_, c_, c1_: tf.less(i, max_cluster_size),
                                                           downdate_params_in_aggregate, loop_vars=state)

        final_all_means = tf.tensor_scatter_nd_update(self.means, tf.expand_dims(non_empty_clusters_set, axis=1),
                                                      final_means)
        final_all_cov_chols = tf.tensor_scatter_nd_update(self.cov_chols,
                                                          tf.expand_dims(non_empty_clusters_set, axis=1),
                                                          final_cov_chols)
        return final_all_means, final_all_cov_chols

    @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)))
    def update_cluster_params_for_batch_in_aggregate(self, sampled_clusters, data_batch, cluster_counts,
                                                     new_old_cluster_mapping):
        def add_indices(curr_clusters_data_points, curr_counts, i, curr_max):
            cluster = unique_clusters_set[i]
            cluster_data_points_indices = tf.cast(tf.reshape(tf.where(sampled_clusters == cluster), [-1]), tf.int32)
            cluster_size = tf.shape(cluster_data_points_indices)[0]
            empty_ = tf.zeros(shape=(self.batch_size_max_value - cluster_size,), dtype=tf.int32) - 1
            # We are filling -1 to cluster data point indices, to have batch_size at the end
            new_ta = curr_clusters_data_points.write(i, tf.concat([cluster_data_points_indices, empty_], axis=0))
            new_max = tf.cond(tf.greater(cluster_size, curr_max), true_fn=lambda: cluster_size,
                              false_fn=lambda: curr_max)

            new_ta1 = tf.cond(cluster == new_cluster_index, true_fn=lambda: curr_counts.write(i, 0.),
                    false_fn=lambda: curr_counts.write(i, cluster_counts[cluster]))
            return new_ta, new_ta1, i + 1, new_max

        def collect_init_params(i, means_ta, cov_chols_ta):
            cluster = unique_clusters_set[i]
            new_means_ta, new_cov_chols_ta = tf.cond(cluster == new_cluster_index,
                                                     true_fn=lambda: (means_ta.write(i, self.mean_0),
                                                                      cov_chols_ta.write(i, self.cov_chol_0)),
                                                     false_fn=lambda: (
                                                     means_ta.write(i, self.means[new_old_cluster_mapping[cluster], :]),
                                                     cov_chols_ta.write(i,
                                                                        self.cov_chols[new_old_cluster_mapping[cluster],
                                                                        :, :])))
            return i + 1, new_means_ta, new_cov_chols_ta

        def update_params_in_aggregate(i, means_, cov_chols_, counts_):
            i_data_points_indices = clusters_data_points_mtx[:, i]
            i_non_empty_indices = tf.reshape(tf.where(i_data_points_indices != -1), [-1])
            i_data_points_indices_valid = tf.gather(i_data_points_indices, i_non_empty_indices)
            i_data_points = tf.gather(data_batch, i_data_points_indices_valid)
            i_means, i_cov_chols = tf.gather(means_, i_non_empty_indices), tf.gather(cov_chols_, i_non_empty_indices)
            i_counts = tf.gather(counts_, i_non_empty_indices)
            new_i_means, new_i_cov_chols = update_(i_data_points, i_means, i_cov_chols, i_counts)
            new_means_ = tf.tensor_scatter_nd_update(means_, tf.expand_dims(i_non_empty_indices, axis=1), new_i_means)
            new_cov_chols_ = tf.tensor_scatter_nd_update(cov_chols_, tf.expand_dims(i_non_empty_indices, axis=1),
                                                         new_i_cov_chols)

            return i + 1, new_means_, new_cov_chols_, counts_ + 1.

        def prepare_indices_for_assignment(i, target_means_, target_cov_chols_, curr_ta):
            cluster = unique_clusters_set[i]
            new_target_means_, new_target_cov_chols_ = tf.cond(cluster == new_cluster_index,
                                                               true_fn=lambda: (
                                                               tf.concat([self.means, [final_means[i, :]]], axis=0),
                                                               tf.concat([self.cov_chols, [final_cov_chols[i, :, :]]],
                                                                         axis=0)),
                                                               false_fn=lambda: (target_means_, target_cov_chols_))

            return tf.cond(cluster == new_cluster_index,
                           true_fn=lambda: (i + 1, new_target_means_, new_target_cov_chols_,
                                            curr_ta.write(i, tf.shape(new_target_means_)[0] - 1)),
                           false_fn=lambda: (i + 1, target_means_, target_cov_chols_,
                                             curr_ta.write(i, new_old_cluster_mapping[cluster])))

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, config.DATA_DIM], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, config.DATA_DIM, config.DATA_DIM], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None])))
        def update_(data_points, means_, cov_chols_, counts_):
            c = tf.expand_dims(self.kappa0 + counts_, axis=1)
            new_means = (means_ * c + data_points) / (c + 1.)
            multipliers = tf.reshape((c + 1.) / c, [-1])

            new_cov_chols = ops.cholesky_update(cov_chols_, data_points - new_means, multiplier=multipliers)
            return new_means, new_cov_chols

        unique_clusters_set, _ = tf.unique(sampled_clusters)
        new_cluster_index = tf.shape(cluster_counts)[0]

        clusters_num = tf.shape(unique_clusters_set)[0]
        clusters_data_points = tf.TensorArray(dtype=tf.int32, size=clusters_num,
                                              element_shape=(self.batch_size_max_value,))
        cluster_counts_ta = tf.TensorArray(dtype=tf.float32, size=clusters_num, element_shape=())
        state = (clusters_data_points, cluster_counts_ta, tf.constant(0), tf.constant(-1))
        filled_clusters_data_points, filled_cluster_counts,  _, max_cluster_size = tf.while_loop(
            lambda ta, ta1, i, m: tf.less(i, clusters_num), add_indices, loop_vars=state)

        clusters_data_points_mtx = filled_clusters_data_points.stack()
        cluster_counts_mtx = filled_cluster_counts.stack()
        m_ta = tf.TensorArray(dtype=tf.float32, size=clusters_num, element_shape=(config.DATA_DIM,))
        c_ta = tf.TensorArray(dtype=tf.float32, size=clusters_num, element_shape=(config.DATA_DIM, config.DATA_DIM))
        _, init_means_ta, init_cov_chols_ta = tf.while_loop(lambda i, m_, c_: tf.less(i, clusters_num),
                                                            collect_init_params, loop_vars=(tf.constant(0), m_ta, c_ta))

        init_means, init_cov_chols = init_means_ta.stack(), init_cov_chols_ta.stack()

        state = (tf.constant(0), init_means, init_cov_chols, cluster_counts_mtx)
        _, final_means, final_cov_chols, _ = tf.while_loop(lambda i, m_, c_, c1_: tf.less(i, max_cluster_size),
                                                           update_params_in_aggregate, loop_vars=state)
        indices_ta = tf.TensorArray(dtype=tf.int32, size=clusters_num, element_shape=())
        state = (tf.constant(0), self.means, self.cov_chols, indices_ta)
        _, t_means, t_cov_chols, filled_indices_ta = tf.while_loop(lambda i, m_, c_, t: tf.less(i, clusters_num),
                                                                   prepare_indices_for_assignment, loop_vars=state)
        old_indices = tf.expand_dims(filled_indices_ta.stack(), axis=1)

        result_means = tf.tensor_scatter_nd_update(t_means, old_indices, final_means)
        result_cov_chols = tf.tensor_scatter_nd_update(t_cov_chols, old_indices, final_cov_chols)

        return result_means, result_cov_chols
