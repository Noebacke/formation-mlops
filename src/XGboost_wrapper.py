import joblib


class ModelWrapper:

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, data):
        return self.model.predict(data)

