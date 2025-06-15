from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import logging
from contextlib import asynccontextmanager
import os

# ----------------------------
# 1. Input schema for prediction
# ----------------------------
class ProductFeatures(BaseModel):
    product_category_name: str
    product_id: str
    qty: int
    freight_price: float
    product_name_lenght: int
    product_description_lenght: int
    product_photos_qty: int
    product_weight_g: float
    product_score: float
    customers: int
    weekday: int
    weekend: int
    holiday: int
    month: int
    year: int
    s: float
    volume: float
    lag_price: float
    fp1: float
    fp2: float
    fp3: float

# ----------------------------
# 2. Model loading/training
# ----------------------------
def load_model():
    global model, feature_columns

    df = pd.read_csv("retail_price.csv")  # Adjust path if needed

    target = 'unit_price'
    categorical_features = ['product_category_name', 'product_id']
    numerical_features = [
        'qty', 'freight_price', 'product_name_lenght', 'product_description_lenght',
        'product_photos_qty', 'product_weight_g', 'product_score', 'customers',
        'weekday', 'weekend', 'holiday', 'month', 'year', 's', 'volume', 'lag_price',
        'fp1', 'fp2', 'fp3'
    ]
    feature_columns = categorical_features + numerical_features

    X = df[feature_columns]
    y = df[target]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    model = pipeline

# ----------------------------
# 3. App with lifespan (startup event)
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App is starting...")
    load_model()
    print("Model loaded.")
    yield
    print("App is shutting down...")

app = FastAPI(
    title="Retail Price Prediction API",
    description="API with HTML frontend for retail price prediction",
    version="1.0",
    lifespan=lifespan
)

# ----------------------------
# 4. Static files + HTML
# ----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("static/index.html") as f:
        return f.read()

# ----------------------------
# 5. Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_price(features: ProductFeatures):
    input_df = pd.DataFrame([features.dict()])
    prediction = model.predict(input_df)[0]
    return JSONResponse(content={"predicted_unit_price": round(float(prediction), 2)})

# ----------------------------
# 6. Optional: Run with Python directly
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
