import mlflow

model_name = "fasttext"
version = 2

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

list_libs = ["vendeur d'huitres", "boulanger", "COIFFEUR"]

results = model.predict(list_libs, params={"k": 1})
print(results)