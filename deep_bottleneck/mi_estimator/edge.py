import numpy as np
import pandas as pd


def load(discretization_range, architecture, n_classes):
    estimator = EDGE(n_classes=n_classes, architecture=architecture)
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

    def hash_entry(self, sample):
        native_hash = hash(tuple(sample))
        result = np.mod(native_hash, self.cHN)
        return result

    def perform_hash(self, x):
        result = map(self.hash_entry, x)
        return list(result)


class EDGE:

    def __init__(self, n_classes, architecture):
        self.n_classes = n_classes
        self.architecture = architecture

    def _initialize_estimator(self, n_examples, n_dimensions):
        self.n_examples = n_examples

        self.cH = 4
        self.epsilon = self.n_examples ** (-1/(2 * n_dimensions))
        self.n_buckets = self.n_examples * self.cH

        self.h1 = H1(self.epsilon)
        self.h2 = H2(self.n_examples, self.cH)

    def _g(self, x):
        result = dict()
        for (x_idx, y_idx), w_ij in x.items():
            result[(x_idx, y_idx)] = x[(x_idx, y_idx)] * np.log(x[(x_idx, y_idx)])
        return result

    def _count_collisions(self, X, Y):

        counts_i = np.zeros(self.n_buckets).astype('int32')
        counts_j = np.zeros(self.n_buckets).astype('int32')
        counts_ij = dict()

        h1_applied_x = self.h1.perform_hash(X)
        h1_applied_y = self.h1.perform_hash(Y)

        h2_applied_x = self.h2.perform_hash(h1_applied_x)
        h2_applied_y = self.h2.perform_hash(h1_applied_y)

        for k in range(self.n_examples):
            h_x = h2_applied_x[k]
            h_y = h2_applied_y[k]
            counts_i[h_x] += 1
            counts_j[h_y] += 1

            if (h_x, h_y) in counts_ij.keys():
                counts_ij[(h_x, h_y)] += 1
            else:
                counts_ij[(h_x, h_y)] = 1

        return counts_i, counts_j, counts_ij

    def _compute_edge_weights(self, counts_i, counts_j, counts_ij):
        w_i = counts_i / self.n_examples
        w_j = counts_j / self.n_examples

        edges = dict()
        for (x_idx, y_idx), c_ij in counts_ij.items():
            edges[(x_idx, y_idx)] = counts_i[x_idx] * counts_j[y_idx]

        w_ij = dict()
        for (x_idx, y_idx), c_ij in counts_ij.items():
            w_ij[(x_idx, y_idx)] = counts_ij[(x_idx, y_idx)] * self.n_examples / edges[(x_idx, y_idx)]

        return w_i, w_j, w_ij

    def _compute_mi_per_epoch_and_layer(self, X, Y):

        counts_i, counts_j, counts_ij = self._count_collisions(X, Y)
        w_i, w_j, w_ij = self._compute_edge_weights(counts_i, counts_j, counts_ij)

        g_applied = self._g(w_ij)

        MI = 0
        for (i_idx, j_idx), w_ij in w_ij.items():
            MI += w_i[i_idx] * w_j[j_idx] * g_applied[(i_idx, j_idx)]

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
        self._initialize_estimator(n_examples=data.examples.shape[0],
                                   n_dimensions=data.examples.shape[1])

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
                layer_activations = np.asarray(layer_activations, dtype=np.float32)

                mi_with_label = self._compute_mi_per_epoch_and_layer(layer_activations, labels[:, np.newaxis])
                mi_with_input = self._compute_mi_per_epoch_and_layer(layer_activations, data.examples)

                measures.loc[(epoch, layer_index), 'MI_XM'] = mi_with_input
                measures.loc[(epoch, layer_index), 'MI_YM'] = mi_with_label
        return measures
