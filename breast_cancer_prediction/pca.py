# Assumption: Data is standardized.


import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        """
        Computes the principal components and the explained variance of the
        given data X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input data.

        Notes
        -----
        The principal components are ordered by their eigen values in
        descending order.
        The explained variance ratio is the ratio of variance explained by each
        principal component to the total variance of the data.

        """
        cov_matrix = np.cov(X.T)
        eigens = np.linalg.eig(cov_matrix)
        eigenvectors = eigens.eigenvectors
        eigenvalues = eigens.eigenvalues

        # max eigenvectors we can retrieve
        self.max_components = eigenvectors.shape[0]

        if self.n_components <= self.max_components:

            # getting top n eigenvalues indices
            idxs = np.argsort(eigenvalues)[::-1]

            # top n eigen vectors-values
            top_eig_vectors = eigenvectors.T[idxs]
            top_eig_values = eigenvalues[idxs]

            # principal components
            self.principal_components_ = top_eig_vectors[: self.n_components].T

            # explained variance ratio
            n_eig_values = top_eig_values[: self.n_components][::-1]
            cumulative_sum_eigvals = np.cumsum(n_eig_values)
            variance_ratio = cumulative_sum_eigvals / np.sum(n_eig_values)
            self.explained_variance_ratio_ = variance_ratio[::-1]

        else:
            raise ValueError(f"n_components must be <= {self.max_components}")

    def transform(self, X):
        """
        Projects the data onto the principal components.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            The projected data.
        """
        # principal components already being transposed during fit method
        return np.dot(X, self.principal_components_)
