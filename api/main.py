import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# Enable CORS for frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://your-render-url.onrender.com"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
MODEL = tf.keras.models.load_model("../saved_models/2")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Path to built React app
build_path = os.path.abspath("../potato-disease-frontend1/build")
app.mount("/static", StaticFiles(directory=os.path.join(build_path, "static")), name="static")

# --- API ROUTES ---
@app.get("/api/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence': confidence
    }

# --- Fallback to React index.html ---
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_file = os.path.join(build_path, "index.html")
    return FileResponse(index_file)
    