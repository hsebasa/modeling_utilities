import unittest
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from statsmodels.genmod.families import Gaussian
from standard_modeling.utilities import StatsModelsRegressor
import warnings


class TestStatsModelsRegressor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = 2 * self.X[:, 0] + 3 * self.X[:, 1] + np.random.randn(100)
        self.sample_weight = np.random.rand(100)

    def test_fit_ols(self):
        model = StatsModelsRegressor(model_class=sm.OLS)
        model.fit(self.X, self.y)
        self.assertTrue(hasattr(model, 'results_'))
        self.assertTrue(hasattr(model, 'coef_'))

    def test_predict_ols(self):
        model = StatsModelsRegressor(model_class=sm.OLS)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_score_ols(self):
        model = StatsModelsRegressor(model_class=sm.OLS)
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        self.assertGreater(score, 0.5)  # Ensuring a reasonable R^2 score

    def test_fit_glm(self):
        model = StatsModelsRegressor(model_class=sm.GLM, family=Gaussian())
        model.fit(self.X, self.y)
        self.assertTrue(hasattr(model, 'results_'))
        self.assertTrue(hasattr(model, 'coef_'))

    def test_offset_handling(self):
        X_with_offset = np.hstack([self.X, self.y.reshape(-1, 1)])
        model = StatsModelsRegressor(model_class=sm.GLM, offset_col=3)
        with warnings.catch_warnings(action="ignore"):
            model.fit(X_with_offset, self.y)
        self.assertTrue(hasattr(model, 'results_'))
        self.assertEqual(len(model.coef_), self.X.shape[1])  # Offset removed from 
        np.testing.assert_array_equal(model.coef_, [0, 0, 0])

    def test_fit_with_sample_weight(self):
        model = StatsModelsRegressor(model_class=sm.WLS)
        model.fit(self.X, self.y, sample_weight=self.sample_weight)
        self.assertTrue(hasattr(model, 'results_'))


if __name__ == '__main__':
    unittest.main()
