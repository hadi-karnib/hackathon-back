from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import mlflow
import mlflow.sklearn
import os

# Import your prediction functions
from Model1 import predict_rate as predict_rate_decision_tree
from Model2 import predict_rate as predict_model2

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost",  # If your frontend is running on localhost
    "http://localhost:3000",  # If your React app runs on port 3000
    "http://your-frontend-domain.com",  # Add your frontend domain here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

def get_latest_version(model_dir: str) -> str:
    try:
        print(f"Checking directory: {model_dir}")
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail=f"Directory does not exist: {model_dir}")
        
        print(f"Contents of the directory: {os.listdir(model_dir)}")
        
        versions = [d for d in os.listdir(model_dir) if d.startswith("version-")]
        if not versions:
            raise HTTPException(status_code=404, detail=f"No versions found in directory: {model_dir}")
        
        print("Versions found:", versions)
        latest_version = sorted(versions, key=lambda x: int(x.split("-")[1]), reverse=True)[0]
        return latest_version.split("-")[1]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not find versions in {model_dir}: {e}")
    

def load_model(model_name: str, version_number: str):
    client = mlflow.tracking.MlflowClient()
    model_uri = f"models:/{model_name}/{version_number}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model version {version_number} not found: {e}")

def load_model1():
    # Use absolute path here
    model_dir = "./mlruns/models/model1-partition3/"
    version_number = get_latest_version(model_dir)
    return load_model("model1-partition3", version_number)

def load_model2():
    # Use absolute path here
    model_dir = "./mlruns/models/model2-partition3/"
    version_number = get_latest_version(model_dir)
    return load_model("model2-partition3", version_number)

class InputDataModel1(BaseModel):
    App: str
    Category: str
    Size_in_MB: int
    Type: str
    Price: float
    Content_Rating: str
    Genres: str
    Last_Updated: str
    Android_Ver: str

class InputDataModel2(BaseModel):
    App: str
    Category: str
    Rating : float
    Reviews: int
    Size_in_MB: int
    Type: str
    Price: float
    Content_Rating: str
    Genres: str
    Last_Updated: str
    Android_Ver: str

@app.post("/predict_decision_tree")
async def predict_decision_tree_endpoint(input_data: InputDataModel1):
    model = load_model1()
    prediction = predict_rate_decision_tree(model, input_data.dict())
    if "error" in prediction:
        raise HTTPException(status_code=400, detail=prediction["error"])
    return prediction

@app.post("/predict_model2")
async def predict_model2_endpoint(input_data: InputDataModel2):
    model = load_model2()
    prediction = predict_model2(model, input_data.dict())
    if "error" in prediction:
        raise HTTPException(status_code=400, detail=prediction["error"])
    return prediction

@app.get("/test")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
