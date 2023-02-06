from datetime import datetime
import os.path as op

import numpy as np
from src.models.clustering import collapsed_gibbs_sampler

from utils import fs_utils


class ParallelCollapsedGibbsSampler(collapsed_gibbs_sampler.CollapsedGibbsSampler):

    def fit(self, iterations_num, data_list):
        assert len(data_list) > 0
        dim = data_list[0].shape[1]

        assert all(data_chunk.shape[1] == dim for data_chunk in data_list)

        from src.models.clustering import config
        config.DATA_DIM = dim
        from src.models.clustering import cgs_utils

        if self.restore_snapshot_pkl_path is not None:
            snapshot = fs_utils.read_pickle(self.restore_snapshot_pkl_path)
            cluster_assignments_list = snapshot['cluster_assignment']
            examples_assignments_list = [self.get_examples_assignment(ca) for ca in cluster_assignments_list]
            alphas_list, init_params_list = snapshot['alpha'], snapshot['cluster_params']
            ass_ll_list = snapshot['ass_ll']
            init_iter = int(op.splitext(op.basename(self.restore_snapshot_pkl_path))[0].split('_')[-1]) + 1
            self.logger.info("Restored params from snapshot path: %s, clusters numbers: %s" %
                             (self.restore_snapshot_pkl_path, str([len(ca) for ca in cluster_assignments_list])))
        else:
            cluster_assignments_list, examples_assignments_list = self.initial_assignment_for_chunks(data_list)
            init_params_list = self.init_params(data_list, cluster_assignments_list)
            alphas_list = [self.alpha for _ in data_list]
            init_iter = 0
            self.logger.info("Initialized first assignments params for data list of length: %d" % len(data_list))
            self.logger.info("Chosen first assignments, clusters num list: %s" %
                             str([len(c) for c in cluster_assignments_list]))

        init_values_list = self.init_values(data_list, cluster_assignments_list, cgs_utils)
        init_values_list = [{**iv, **{'means': ip['mean'], 'cov_chols': ip['cov_chol']}}
                            for iv, ip in zip(init_values_list, init_params_list)]
        nus_list = [v['nu_0'] for v in init_values_list]
        n_points_list = [data_chunk.shape[0] for data_chunk in data_list]
        computation_managers_list = [cgs_utils.TfCgsSharedComputationsManager(data_chunk, len(ca), init_values,
                                                                              self.batch_size)
                                     for data_chunk, ca, init_values in zip(data_list, cluster_assignments_list,
                                                                            init_values_list)]
        ex_permutation_list = [list(range(data_chunk.shape[0])) for data_chunk in data_list]
        permutation_matrix = self.gen_permutation_matrix(ex_permutation_list)
        for iter_num in range(init_iter, iterations_num, 1):
            if iter_num == init_iter:
                curr_params_list = init_params_list

            if iter_num % self.skip_epochs_ll_calc == 0:
                ass_ll_list = self.calculate_lls(cluster_assignments_list, data_list, curr_params_list, nus_list)
                self.logger.info("Epoch: %d, clusters counts: %s, assignment lls: %s" %
                                 (iter_num, str([len(ca) for ca in cluster_assignments_list]), ass_ll_list))

            last_log_counter = 0
            iter_state = {'alphas': alphas_list, 'res': [None] * len(n_points_list)}
            for perm_column in range(0, permutation_matrix.shape[1], self.batch_size):
                data_list_perm_indices = permutation_matrix[:, perm_column: perm_column + self.batch_size]
                # perm_column_results, alphas_list =
                self.do_iteration_for_data_list(cluster_assignments_list, examples_assignments_list,
                                                data_list_perm_indices, computation_managers_list, n_points_list,
                                                iter_state)

                if int(perm_column / 100) > last_log_counter:
                    last_log_counter = int(perm_column / 100)
                    timestamp_str = datetime.now().strftime("%H:%M:%S")
                    self.logger.info(
                        "Epoch: %d, examples processed for all data chunks: %d/%d, clusters counts: %s, %s" %
                        (iter_num, perm_column + data_list_perm_indices.shape[1],
                         permutation_matrix.shape[1], str([len(ca) for ca in cluster_assignments_list]),
                         timestamp_str))

            curr_params_list = iter_state['res']
            if iter_num % self.skip_epochs_logging == 0:
                self.save_model(curr_params_list, cluster_assignments_list, data_list, iter_num, alphas_list,
                                ass_ll_list)
                self.logger.info("Saved model from iteration: %d" % iter_num)

            self.shuffle_permutation_matrix(permutation_matrix, n_points_list)

        return curr_params_list

    def do_iteration_for_data_list(self, cluster_assignment_list, examples_assignment_list,
                                data_list_perm_indices, computation_managers_list, n_points_list, iter_state):

        for i, curr_alpha in zip(range(data_list_perm_indices.shape[0]), iter_state['alphas']):
            cluster_assignment, examples_assignment = cluster_assignment_list[i], examples_assignment_list[i]
            data_points_indices = data_list_perm_indices[i, :]
            data_points_indices_filtered = [i for i in data_points_indices if i != -1]
            if len(data_points_indices_filtered) == 0:
                continue

            n_points = n_points_list[i]
            computations_manager = computation_managers_list[i]

            upd_alpha = self.update_alpha(curr_alpha, n_points, len(cluster_assignment))
            iter_state['alphas'][i] = upd_alpha

            cluster_counts_before_removal = [len(cluster_examples) for cluster_examples in cluster_assignment]
            data_batch_clusters, clusters_batch_removed = self.remove_assignment_for_batch(data_points_indices_filtered,
                                                                                           cluster_assignment,
                                                                                           examples_assignment)

            if data_points_indices[-1] == data_list_perm_indices[i, -1] or data_points_indices[-1] == -1:
                return_curr_params = True
            else:
                return_curr_params = False

            curr_cluster_counts = [len(cluster_examples) for cluster_examples in cluster_assignment]
            res = computations_manager.sample_clusters_for_data_batch(data_points_indices_filtered, data_batch_clusters,
                                                                      clusters_batch_removed,
                                                                      cluster_counts_before_removal,
                                                                      curr_cluster_counts, curr_alpha,
                                                                      return_curr_params=return_curr_params)
            z_indices = res['sampled_clusters']
            new_cluster_index = len(cluster_assignment)
            z_indices = [new_cluster_index if z_indx > new_cluster_index else z_indx for z_indx in z_indices]
            for z_indx, data_point_indx in zip(z_indices, data_points_indices_filtered):
                if z_indx == new_cluster_index:
                    if len(cluster_assignment) == new_cluster_index:
                        cluster_assignment.append({data_point_indx})
                    else:
                        # len(cluster_assignment) > new_cluster_index
                        cluster_assignment[z_indx].add(data_point_indx)
                else:
                    cluster_assignment[z_indx].add(data_point_indx)

                examples_assignment[data_point_indx] = z_indx

            iter_state['res'][i] = res

    @staticmethod
    def shuffle_permutation_matrix(permutation_matrix, data_chunks_sizes):
        for i, data_chunk_size in zip(range(permutation_matrix.shape[0]), data_chunks_sizes):
            np.random.shuffle(permutation_matrix[i, :data_chunk_size])

    def gen_permutation_matrix(self, init_permutation_list):
        rows = []
        biggest_data_chunk_size = max([len(perm) for perm in init_permutation_list])
        for data_chunk_permutation in init_permutation_list:
            data_chunk_size = len(data_chunk_permutation)
            data_chunk_permutation_ext = data_chunk_permutation + [-1] * (biggest_data_chunk_size - data_chunk_size)

            rows.append(data_chunk_permutation_ext)

        result = np.array(rows, dtype=np.int32)
        self.logger.info("Generated initial permutation matrix (shape: %s) of num of columns equal "
                         "to biggest data chunk: %d" % (str(result.shape), biggest_data_chunk_size))

        return result

    def calculate_lls(self, cluster_assignment_list, data_list, curr_params_list, nus_list):
        ass_ll_list = []
        for cluster_assignment, data_chunk, curr_params, nu_0 in zip(cluster_assignment_list, data_list,
                                                                     curr_params_list, nus_list):
            ass_ll = self.data_log_likelihood(cluster_assignment, data_chunk, curr_params, nu_0)
            ass_ll_list.append(ass_ll)

        return ass_ll_list

    def initial_assignment_for_chunks(self, data_list):
        cluster_assignments_list, examples_assignments_list = [], []
        for data_chunk in data_list:
            chunk_ca, chunk_ea = self.get_initial_assignment(data_chunk)
            cluster_assignments_list.append(chunk_ca)
            examples_assignments_list.append(chunk_ea)

        return cluster_assignments_list, examples_assignments_list

    def init_values(self, data_list, cluster_assignments_list, utils_module):
        n_comps = [len(ca) for ca in cluster_assignments_list]
        mean_0_list, cov_0_list = [], []
        for data_chunk, n in zip(data_list, n_comps):
            mean_0, cov_0 = self.initial_mean0_cov0(data_chunk, data_chunk, n)
            mean_0_list.append(mean_0)
            cov_0_list.append(cov_0)

        nu_0_list = [utils_module.init_nu_0(data_chunk) for data_chunk in data_list]
        cov_chol_0_list = [np.linalg.cholesky(cov_0) for cov_0 in cov_0_list]
        result = []
        for mean_0, cov_chol_0, nu_0 in zip(mean_0_list, cov_chol_0_list, nu_0_list):
            result.append({'mean_0': mean_0, 'cov_chol_0': cov_chol_0, 'nu_0': nu_0})

        return result

    def init_params(self, data_list, cluster_assignments_list):
        n_comps = [len(ca) for ca in cluster_assignments_list]
        init_cluster_params_list = [self.assign_initial_params(ca, data_chunk, n) for ca, data_chunk, n in
                                    zip(cluster_assignments_list, data_list, n_comps)]
        return [{'mean': np.array(init_params['mean']), 'cov_chol': np.array(init_params['cov_chol'])}
                for init_params in init_cluster_params_list]
