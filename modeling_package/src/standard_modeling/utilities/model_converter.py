from sklearn.base import BaseEstimator, RegressorMixin

import statsmodels.api as sm

import pandas as pd
import numpy as np

import warnings

from .scoring import gen_scorer


class StatsModelsRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible wrapper for statsmodels regression models.
    
    Parameters
    ----------
    model_class : statsmodels model class, default=sm.OLS
        The statsmodels model class to use (e.g., sm.OLS, sm.GLM).
    fit_intercept : bool, default=True
        Whether to fit an intercept term.
    alpha : float, optional
        L2 regularization term for regularized models (if applicable).
    family : statsmodels.genmod.families.Family, optional
        The distribution family for GLM models.
    scoring : str, default='r2'
        The scoring metric used for evaluation.
    offset_col : int, optional
        Index of the column in X to use as an offset term (removes from training features).
    fit_kwargs : dict, optional
        Additional keyword arguments passed to the model's fit method.
    **kwargs : additional parameters
        Extra parameters to pass to the statsmodels model.
    
    Attributes
    ----------
    model_ : statsmodels model instance
        The fitted statsmodels model.
    results_ : statsmodels results instance
        The results from fitting the model.
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Intercept term.
    """
    
    def __init__(self, model_class=sm.OLS, fit_intercept=True, alpha=None, 
                 family=None, scoring='r2', offset_col=None, fit_kwargs=None, **kwargs):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.family = family
        self.offset_col = offset_col
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.kwargs = kwargs
        self.scoring = scoring

    def _preprocess(self, X):
        offset = None
        if self.offset_col is not None:
            offset = X[:, self.offset_col]
            X = np.delete(X, self.offset_col, axis=1)
        
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
        else:
            X = X
        return offset, X
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        offset, X_model = self._preprocess(X)
        
        model_kwargs = self.kwargs.copy()

        if issubclass(self.model_class, sm.GLM) and self.family is not None:
            model_kwargs['family'] = self.family

        if sample_weight is not None:
            model_kwargs['weights'] = sample_weight
        
        if offset is not None:
            model_kwargs['offset'] = offset

        self.model_ = self.model_class(y, X_model, **model_kwargs)

        if self.alpha is not None and hasattr(self.model_, "fit_regularized"):
            self.results_ = self.model_.fit_regularized(alpha=self.alpha, **self.fit_kwargs)
        else:
            self.results_ = self.model_.fit(**self.fit_kwargs)

        if self.fit_intercept:
            self.intercept_ = self.results_.params[0]
            self.coef_ = self.results_.params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.results_.params

        return self

    def predict(self, X):
        """
        Generate predictions using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X)
        
        offset, X_pred = self._preprocess(X)
        
        return self.results_.predict(X_pred)
    
    def score(self, X, y, sample_weight=None, **kwargs):
        """
        Compute the score of the model (default: R^2 score).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        
        Returns
        -------
        score : float
            The score of the model.
        """
        X = np.asarray(X)
        y_pred = self.predict(X)
        scorer = gen_scorer(self.scoring)
        return scorer(y, y_pred, sample_weight=sample_weight, **kwargs)
    
    @property
    def summary(self):
        """
        Return the summary output from statsmodels.
        
        Returns
        -------
        summary : statsmodels.iolib.summary.Summary
            Summary of the fitted model.
        """
        if hasattr(self, 'results_'):
            return self.results_.summary()
        else:
            raise AttributeError("Model not fitted. Call 'fit' first.")
