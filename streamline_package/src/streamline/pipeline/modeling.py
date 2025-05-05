
class BaseModel:
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    def score(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class BaseTransformer:
    def transform(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    
    
