import os
from OrangeClass import OrangeDetectorAndClassifier_v2
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Get the directory of the current script (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the project root, then into the models directory
model_dir = os.path.normpath(os.path.join(current_dir, '../../models'))

# Updated paths
YOLO_ORANGE_DETECTION_COUNTING_MODEL = os.path.join(model_dir, "Orange detection counting.pt")
OUTPUT_JSON_FILE_PATH = os.path.join(current_dir, "output.json")
CLASS_NAMES = ["Blood Oranges", "Navel", "Tangelo", "Tangerine", "Cara Cara"]
CLASSIFICATION_MODEL_TYPE = "vit"  # or "efficientnet"

DETECTRON_CONFIG_YAML_FILE_PATH = os.path.join(
    model_dir, "Detectron 2 Orange dieases segmentation", "config.yaml"
)
DETECTRON_DIEASE_SEGMENTATION_MODEL_PATH = os.path.join(
    model_dir, "Detectron 2 Orange dieases segmentation", "model_final.pth"
)

# Select the correct classification model path based on the model type
if CLASSIFICATION_MODEL_TYPE == "vit":
    CLASSIFICATION_MODEL_PATH = os.path.join(model_dir, "vit_classification_model.pth")
else:
    CLASSIFICATION_MODEL_PATH = os.path.join(model_dir, "efficientnet_best.pth")


## initialise the classs 
orange_processor = OrangeDetectorAndClassifier_v2(YOLO_ORANGE_DETECTION_COUNTING_MODEL, CLASSIFICATION_MODEL_TYPE, CLASS_NAMES, CLASSIFICATION_MODEL_PATH)
'''
## use method for using pipeline (detecion and classification of oranges)
result = orange_processor.process_image(image_path, OUTPUT_JSON_FILE_PATH)

## use method to count instance of orange images
b_boxes, detection_time , count_of_oranges = orange_processor.detect_oranges(image_path)

## given the image return the type/species of oranges from 5 
img = Image.open(image_path)
orange_type = orange_processor.classify_orange(img,"vit")'''


# Route for processing image
@app.post("/process/")
async def process_image(image: UploadFile = File(...), output_json_path: str = Form(...)):
    image_path = f"temp_images/{image.filename}"
    
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = orange_processor.process_image(image_path, output_json_path)
    
    # Remove the temporary image
    os.remove(image_path)
    
    return JSONResponse(content=result)

# Route for detecting oranges
@app.post("/detect/")
async def detect_oranges(image: UploadFile = File(...)):
    image_path = f"temp_images/{image.filename}"
    
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    b_boxes,time_required,number_of_oranges = orange_processor.detect_oranges(image_path)
    
    # Remove the temporary image
    os.remove(image_path)
    
    return JSONResponse(content=f"the number of oranges detected in image {number_of_oranges}")

# Route for classifying oranges
@app.post("/classify/")
async def classify_orange(image: UploadFile = File(...), modelType: str = Form(...)):
    image_path = f"temp_images/{image.filename}"
    
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    img = Image.open(image_path)
    orange_type = orange_processor.classify_orange(img, modelType)
    
    # Remove the temporary image
    os.remove(image_path)
    
    return JSONResponse(content={"orange_type": orange_type})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)