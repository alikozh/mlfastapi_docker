# 1. Python base image
FROM python:3.12-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Install dependencies first (this layer is cached if requirements.txt doesn't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the project files
COPY . .

# 5. Re-train the model inside the image so model.joblib matches
#    the scikit-learn version installed in the container
RUN python train.py

# 6. Run the FastAPI app with Uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
