import numpy as np
import pandas as pd


def load(discretization_range, architecture, n_classes):
    estimator = EDGE(3277, n_classes=n_classes, architecture=architecture)
    return estimator


class H1:

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.b = np.random.uniform(0.0, epsilon)

    def perform_hash(self, x):
        return np.floor((x + self.b) / self.epsilon).astype('int32')


class H2:

    def __init__(self, N, cH):
        self.cHN = N * cH

    def perform_hash(self, sample):
        str_sample = str(sample)
        native_hash = hash(str_sample)
        result = np.mod(native_hash, self.cHN)

        return result


class EDGE:

    def __init__(self, n_examples, n_classes, architecture):
        self.n_examples = n_examples
        self.n_classes = n_classes
        self.architecture = architecture

        self.cH = 4
        self.epsilon = 0.008
        self.n_buckets = self.n_examples * self.cH

        self.h1 = H1(self.epsilon)
        self.h2 = H2(self.n_examples, self.cH)

    def _g(self, x):
        result = np.zeros_like(x).astype('float32')
        result[x != 0.0] = x[x != 0.0] * np.log(x[x != 0.0])
        return result

    def _count_collisions(self, X, Y):

        counts_i = np.zeros(self.n_buckets).astype('int32')
        counts_j = np.zeros(self.n_buckets).astype('int32')
        counts_ij = np.zeros((self.n_buckets, self.n_buckets)).astype('int32')
        for k in range(self.n_examples):
            h_x = self.h2.perform_hash(self.h1.perform_hash(X[k]))
            h_y = self.h2.perform_hash(self.h1.perform_hash(Y[k]))
            counts_i[h_x] += 1
            counts_j[h_y] += 1
            counts_ij[h_x, h_y] += 1

        return counts_i, counts_j, counts_ij

    def _compute_edge_weights(self, counts_i, counts_j, counts_ij):
        w_i = counts_i / self.n_examples
        w_j = counts_j / self.n_examples

        # this will cause division by zero warnings
        w_ij = counts_ij * self.n_examples / (counts_i * counts_j)
        w_ij[np.isinf(w_ij)] = 0  # workaround
        w_ij[np.isnan(w_ij)] = 0  # workaround

        return w_i, w_j, w_ij

    def _compute_mi_per_epoch_and_layer(self, X, Y):
        #print("X:" + str(X.shape))
        #print("Y:" + str(Y.shape))

        counts_i, counts_j, counts_ij = self._count_collisions(X, Y)
        w_i, w_j, w_ij = self._compute_edge_weights(counts_i, counts_j, counts_ij)

        g_applied = self._g(w_ij)
        # lower bound # used bins for Y
        used_bins_y = np.sum(counts_j[counts_j != 0])
        U = np.ones_like(g_applied) * used_bins_y

        stacked = np.stack([g_applied, U])
        g_schlange = np.max(stacked, axis=0)

        nonzero = np.nonzero(w_ij)
        MI = 0
        for idx in range(len(nonzero[0])):
            i_idx = nonzero[0][idx]
            j_idx = nonzero[1][idx]
            MI += w_i[i_idx] * w_j[j_idx] * g_schlange[i_idx, j_idx]

        return MI

    def _init_dataframe(self, epoch_numbers, n_layers):
        info_measures = ['MI_XM', 'MI_YM']
        index_base_keys = [epoch_numbers, list(range(n_layers))]
        index = pd.MultiIndex.from_product(index_base_keys, names=['epoch', 'layer'])
        measures = pd.DataFrame(index=index, columns=info_measures)
        return measures

    def compute_mi(self, data, file_dump) -> pd.DataFrame:
        print(f'*** Start running {self.__class__.__name__}. ***')

        labels = data.labels
        one_hot_labels = data.one_hot_labels

        n_layers = len(self.architecture) + 1  # + 1 for output layer
        epoch_numbers = [int(value) for value in file_dump.keys()]
        epoch_numbers = sorted(epoch_numbers)
        measures = self._init_dataframe(epoch_numbers=epoch_numbers, n_layers=n_layers)

        for epoch in epoch_numbers:
            print(f'Estimating mutual information for epoch {epoch}.')
            summary = file_dump[str(epoch)]
            for layer_index in range(n_layers):
                layer_activations = summary['activations'][str(layer_index)]
                mi_with_label = self._compute_mi_per_epoch_and_layer(layer_activations, labels)
                mi_with_input = self._compute_mi_per_epoch_and_layer(layer_activations, data.examples)
                print(mi_with_label)
                print(mi_with_input)
                measures.loc[(epoch, layer_index), 'MI_XM'] = mi_with_input
                measures.loc[(epoch, layer_index), 'MI_YM'] = mi_with_label
        return measures
