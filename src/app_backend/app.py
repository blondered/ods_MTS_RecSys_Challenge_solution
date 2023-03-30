from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import mlflow
import os

load_dotenv()

class Model:
    def __init__(self, model_name, model_stage):
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions

model = Model("kion_catboost", "Production")

app = FastAPI()

@app.get("/hello-world")
def root():
    return {"message": "Hello, world!"}

@app.post("/invocations")
async def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        return list(model.predict(data))
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted")

# Check environmet variables for AWS access
if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None:
    exit(1)
