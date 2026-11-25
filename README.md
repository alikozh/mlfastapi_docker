# ML Model Deployment with FastAPI and Docker

A Random Forest classifier trained on the classic **Iris** dataset, served as a REST API with **FastAPI** and packaged in a **Docker** container.

The model predicts the Iris species (`setosa`, `versicolor`, `virginica`) from four flower measurements in cm.

## Project structure

```
ml-fastapi-docker/
├── main.py            # FastAPI app (GET /, POST /predict)
├── train.py           # trains the model and saves model.joblib
├── model.joblib       # saved model artifact
├── requirements.txt   # Python dependencies
├── Dockerfile         # container instructions
└── README.md
```

## 1. Train the model

```bash
pip install -r requirements.txt
python train.py
```

This prints the test accuracy (about 0.90) and saves `model.joblib`.
The file stores the trained model together with the class names, and the API loads it at startup.

## 2. Run the API locally (Uvicorn)

```bash
uvicorn main:app --reload
```

Then check:

| What | URL |
|---|---|
| Root endpoint | http://127.0.0.1:8000/ → `{"message": "ML API is running"}` |
| Swagger docs | http://127.0.0.1:8000/docs |

### Test `/predict`

In Swagger: open **POST /predict** → **Try it out** → **Execute**.

Or with curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}
}
```

Invalid input (a missing field, text instead of a number, or a value ≤ 0) returns a `422` validation error.

## 3. Build the Docker image

```bash
docker build -t ml-fastapi-app .
```

The Dockerfile uses a `python:3.12-slim` base image, sets `/app` as the working directory, installs the dependencies, copies the project, re-trains the model inside the image (so `model.joblib` matches the container's scikit-learn version) and starts Uvicorn.

## 4. Run the container

```bash
docker run -d -p 8000:8000 --name ml-api ml-fastapi-app
```

Test again at http://127.0.0.1:8000/, http://127.0.0.1:8000/docs and with the curl command above.

Stop and remove the container:

```bash
docker stop ml-api && docker rm ml-api
```
