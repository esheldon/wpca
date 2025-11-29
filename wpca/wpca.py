import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from .utils import check_array_with_weights, weighted_mean


class WPCA(BaseEstimator, TransformerMixin):
    """Weighted Principal Component Analysis

    This is a direct implementation of weighted PCA based on the eigenvalue
    decomposition of the weighted covariance matrix following
    Delchambre (2014) [1]_.

    Parameters
    ----------
    n_components : int (optional)
        Number of components to keep. If not specified, all components are kept

    xi : float (optional)
        Degree of weight enhancement.

    regularization : float (optional)
        Control the strength of ridge regularization used to compute the
        transform.

    copy_data : boolean, optional, default True
        If True, X and weights will be copied; else, they may be overwritten.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.

    See Also
    --------
    - PCA
    - sklearn.decomposition.PCA

    References
    ----------
    .. [1] Delchambre, L. MNRAS 2014 446 (2): 3545-3555 (2014)
           http://arxiv.org/abs/1412.4533
    """

    def __init__(self, n_components=None, xi=0, regularization=None, copy_data=True):
        self.n_components = n_components
        self.xi = xi
        self.regularization = regularization
        self.copy_data = copy_data

    def save(self, fname, meta=None):
        import fitsio

        header = {}
        if meta is not None:
            header.update(meta)

        header["n_components"] = self.n_components
        header["xi"] = self.xi
        header["copy_data"] = self.copy_data
        header["n_iter"] = self.n_iter_

        dtype = [
            ("mean", "f8", self.mean_.shape),
            ("explained_variance", "f8", self.explained_variance_.shape),
            ("explained_variance_ratio", "f8", self.explained_variance_ratio_.shape),
            ("components", "f8", self.components_.shape[1:]),
        ]

        try:
            len(self.regularization)
            dtype += [
                ("regularization", self.regularization.shape),
            ]
            reg_is_array = True
        except TypeError:
            if self.regularization is not None:
                header["regularization"] = self.regularization
            reg_is_array = False

        with fitsio.FITS(fname, "rw", clobber=True) as fits:
            fits.write(self.mean_, extname="mean", header=header)
            fits.write(
                self.explained_variance_, extname="explained_variance", header=header
            )
            fits.write(
                self.explained_variance_ratio_,
                extname="explained_variance_ratio",
                header=header,
            )
            fits.write(self.components_, extname="components", header=header)
            if reg_is_array:
                fits.write(self.regularization, extname="regularization", header=header)

    def load(self, fname):
        import fitsio

        with fitsio.FITS(fname) as fits:
            header = fits["mean"].read_header()

            self.mean_ = fits["mean"].read()
            self.explained_variance_ = fits["explained_variance"].read()
            self.explained_variance_ratio_ = fits["explained_variance_ratio"].read()
            self.components_ = fits["components"].read()

            if "regularization" in fits:
                self.regularization = fits["regularization"].read()
            else:
                self.regularization = header.get("regularization")

        self.n_components = self.components_.shape[0]
        self.xi = header["xi"]
        self.copy_data = header["copy_data"]
        self.n_iter = header["n_iter"]

    @classmethod
    def fromfile(cls, fname):
        pca = WPCA()
        pca.load(fname)
        return pca

    def _center_and_weight(self, X, weights, fit_mean=False, keepnone=False):
        """Compute centered and weighted version of X and adjust weights.

        Input weights are inverse variance and adjusted weights are
        inverse sigmas. Will produce a RuntimeWarning if any input
        weights are negative.

        If fit_mean is True, then also save the mean to self.mean_
        """
        X, weights = check_array_with_weights(
            X, weights, dtype=float, copy=self.copy_data
        )

        if fit_mean:
            self.mean_ = weighted_mean(X, weights, axis=0)

        # now let X <- (X - mean) * weights
        X -= self.mean_

        if weights is not None:
            # Convert from inverse variance to inverse sigmas.
            # See eqn. 7 of Delchambre 2015 and issue #2.
            weights = np.sqrt(weights.clip(min=0))
            X *= weights
        else:
            if not keepnone:
                weights = np.ones_like(X)

        return X, weights

    def fit(self, X, y=None, weights=None):
        """Compute principal components for X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # let X <- (X - mean) * weights
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_precentered(X, weights)
        return self

    def _fit_precentered(self, X, weights):
        """fit pre-centered data"""
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        # TODO: filter NaN warnings
        covar = np.dot(X.T, X)
        covar /= np.dot(weights.T, weights)
        covar[np.isnan(covar)] = 0

        # enhance weights if desired
        if self.xi != 0:
            Ws = weights.sum(0)
            covar *= np.outer(Ws, Ws) ** self.xi

        subset_by_index = (X.shape[1] - n_components, X.shape[1] - 1)
        evals, evecs = linalg.eigh(covar, subset_by_index=subset_by_index)
        self.components_ = evecs[:, ::-1].T
        self.explained_variance_ = evals[::-1]
        self.explained_variance_ratio_ = evals[::-1] / covar.trace()

        self.n_iter_ = 1  # needed by sklearn.utils.estimator_checks

    def transform(self, X, weights=None, progress=False):
        """Apply dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse variance when errors are Gaussian.

        progress: bool, optional
            If set to True, show a progress bar

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=False, keepnone=True)
        return self._transform_precentered(X, weights, progress=progress)

    def _transform_precentered(self, X, weights, progress=False):
        """
        transform pre-centered data

                tdot1: 1.2636184692382812e-05
                tdot2: 0.00012993812561035156
                tsolve: 2.4557113647460938e-05
                tsub: 0.00016736984252929688
                tot: 0.00016808509826660156

        """
        if progress:
            from tqdm import trange
            from functools import partial
            miter = partial(trange, ascii=True, ncols=70)
        else:
            miter = range

        # import time

        # TODO: parallelize this?
        Y = np.zeros((X.shape[0], self.components_.shape[0]))
        # ttot = 0.0
        # tdot1 = 0.0
        # tdot2 = 0.0
        # tmult = 0.0
        # tsolve = 0.0

        if weights is None:
            cW = self.components_
        # ttot_tm0 = time.time()
        for i in miter(X.shape[0]):
            # tm0 = time.time()
            if weights is not None:
                cW = self.components_ * weights[i]
            # tmult += time.time() - tm0

            # tm0 = time.time()
            cWX = np.dot(cW, X[i])
            # tdot1 += time.time() - tm0
            # tm0 = time.time()
            cWc = np.dot(cW, cW.T)
            # import IPython; IPython.embed()
            # tdot2 += time.time() - tm0

            if self.regularization is not None:
                cWc += np.diag(self.regularization / self.explained_variance_)
            # tm0 = time.time()
            Y[i] = np.linalg.solve(cWc, cWX)
            # tsolve += time.time() - tm0
        # ttot = time.time() - ttot_tm0
        # print('----------')
        # print('tmult:', tmult)
        # print('tdot1:', tdot1)
        # print('tdot2:', tdot2)
        # print('tsolve:', tsolve)
        # print('tsub:', tmult+tdot1+tdot2+tsolve)
        # print('tot:', ttot)
        return Y

    def fit_transform(self, X, y=None, weights=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X, weights = self._center_and_weight(X, weights, fit_mean=True)
        self._fit_precentered(X, weights)
        return self._transform_precentered(X, weights)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
        """
        X = check_array(X)
        return self.mean_ + np.dot(X, self.components_)

    def reconstruct(self, X, weights=None):
        """Reconstruct the data using the PCA model

        This is equivalent to calling transform followed by inverse_transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.transform(X, weights=weights))

    def fit_reconstruct(self, X, weights=None):
        """Fit the model and reconstruct the data using the PCA model

        This is equivalent to calling fit_transform()
        followed by inverse_transform().

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed representation.

        weights: array-like, shape (n_samples, n_features)
            Non-negative weights encoding the reliability of each measurement.
            Equivalent to the inverse variance when errors are Gaussian.

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_components)
            Reconstructed version of X
        """
        return self.inverse_transform(self.fit_transform(X, weights=weights))
