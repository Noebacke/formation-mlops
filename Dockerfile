FROM daskdev/dask-xgboost:latest

# Installer les dépendances Python supplémentaires
RUN pip install fastapi uvicorn mlflow lz4 psutil

# Copier votre application dans l'image Docker
COPY . /app
WORKDIR /app

# launch the unicorn server to run the api
EXPOSE 8000
CMD ["uvicorn", "app.main:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
