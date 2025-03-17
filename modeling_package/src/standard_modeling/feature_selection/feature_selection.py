from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.metrics import get_scorer
from joblib import Parallel, delayed
import pandas as pd
import numpy as np


def cross_validate_model(model, X, y, cv=5, scoring='accuracy', fit_kwargs=None, score_kwargs=None, sample_weights=None, random_state=None):
    """
    Perform cross-validation on a given model without using cross_val_score.
    """
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    scorer = get_scorer(scoring)
    scores = []
    fit_kwargs = fit_kwargs if fit_kwargs else {}
    score_kwargs = score_kwargs if score_kwargs else {}
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        fit_kwargs_with_weights = fit_kwargs.copy()
        if sample_weights is not None:
            fit_kwargs_with_weights['sample_weight'] = sample_weights[train_index]
        
        model.fit(X_train, y_train, **fit_kwargs_with_weights)
        score = scorer(model, X_test, y_test, **score_kwargs)
        scores.append(score)
    
    return scores

class FeatureSelectionResult:
    def __init__(self, selected_features, scores, vif_values, final_model, final_score):
        """
        Store the results of feature selection.
        """
        self.selected_features = selected_features
        self.scores = scores
        self.vif_values = vif_values
        self.final_model = final_model
        self.final_score = final_score
        
        # Create a DataFrame for better readability
        self.feature_summary = pd.DataFrame({
            'Feature': self.selected_features,
            'Score': [self.scores[a] for a in self.selected_features],
            'VIF': [self.vif_values[a] for a in self.selected_features]
        })

    def get_summary(self):
        return self.feature_summary


class FeatureSelector:
    def __init__(self, cv=5, scoring='accuracy', vif_threshold=5, score_threshold=0.01,
                 max_features=None, fit_kwargs=None, score_kwargs=None, parallel_kwargs=None,
                 cross_val_kwargs=None):
        """
        Initialize the FeatureSelector with common parameters.
        """
        self.cv = cv
        self.scoring = scoring
        self.vif_threshold = vif_threshold
        self.score_threshold = score_threshold
        self.max_features = max_features
        self.fit_kwargs = fit_kwargs if fit_kwargs else {}
        self.score_kwargs = score_kwargs if score_kwargs else {}
        self.parallel_kwargs = parallel_kwargs if parallel_kwargs else {}
        self.cross_val_kwargs = cross_val_kwargs if cross_val_kwargs else {}

    def _evaluate_feature(self, model, X, y, selected_features, feature, sample_weights=None):
        """
        Evaluate a feature by computing VIF and cross-validation score.
        """
        candidate_features = selected_features + [feature]
        X_subset = X[candidate_features]

        if X_subset.shape[1] > 1:
            # Compute VIF
            vif = [variance_inflation_factor(X_subset.values, i) for i in range(X_subset.shape[1])]
            max_vif = max(vif)
        else:
            max_vif = 1

        # Evaluate model performance using cross-validation
        score = np.mean(cross_validate_model(model, X_subset, y, cv=self.cv, scoring=self.scoring, fit_kwargs=self.fit_kwargs,
                                             score_kwargs=self.score_kwargs, sample_weights=sample_weights, **self.cross_val_kwargs))
        
        return feature, score, max_vif

    def forward_feature_selection(self, model, X, y, initial_features=None, exclude_features=None, sample_weights=None):
        """
        Forward feature selection: Adds features one at a time based on model performance and VIF constraints.
        """
        selected_features = initial_features.copy() if initial_features else []
        exclude_features = exclude_features if exclude_features else []
        remaining_features = list(set(X.columns) - set(selected_features) - set(exclude_features))
        scores = {}
        vif_values = {}
        
        if selected_features:
            prev_score = np.mean(cross_validate_model(model, X[selected_features], y, cv=self.cv, scoring=self.scoring,
                fit_kwargs=self.fit_kwargs, score_kwargs=self.score_kwargs, sample_weights=sample_weights, **self.cross_val_kwargs
            ))
        else:
            prev_score = -np.inf
        
        while remaining_features:
            if self.max_features and len(selected_features) >= self.max_features:
                break  # Stop if max feature limit is reached

            best_feature = None
            best_score = prev_score
            best_vif = None

            results = Parallel(**self.parallel_kwargs)(
                delayed(self._evaluate_feature)(model, X, y, selected_features, feature, sample_weights)
                for feature in remaining_features
            )

            for feature, score, vif in results:
                if vif < self.vif_threshold and score - prev_score > self.score_threshold and score > best_score:
                    best_score = score
                    best_feature = feature
                    best_vif = vif

            if best_feature:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                scores[best_feature] = best_score
                vif_values[best_feature] = best_vif
                prev_score = best_score  # Update prev_score without recalculating
            else:
                break  # Stop if no feature meets the criteria

        # Train final model with selected features
        fit_kwargs_with_weights = self.fit_kwargs.copy()
        if sample_weights is not None:
            fit_kwargs_with_weights['sample_weight'] = sample_weights
        final_model = model.fit(X[selected_features], y, **fit_kwargs_with_weights)
        scorer = get_scorer(self.scoring)
        final_score = scorer(final_model, X[selected_features], y, **self.score_kwargs)

        return FeatureSelectionResult(selected_features, scores, vif_values, final_model, final_score)
