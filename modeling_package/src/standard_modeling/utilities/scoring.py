from sklearn.metrics import get_scorer


def gen_scorer(scoring):
    scorer = get_scorer(scoring)
    class DummyEstimator:
        def __init__(self, y_pred, **kwargs):
            self.y_pred = y_pred
            
        def predict(self, X, **kwargs):
            return self.y_pred
    
    def score(y_true, y_pred, *args, **kwargs):
        return scorer(DummyEstimator(y_pred), None, y_true, *args, **kwargs)
        
    return score
        