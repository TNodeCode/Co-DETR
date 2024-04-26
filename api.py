import os
import time
import torch
import yaml
from mmdet.apis import inference_detector, init_detector
from fastapi import FastAPI, UploadFile, File



device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE", device)

available_models = {}
current_model_id = ""
current_model = None

def get_available_models():
    global available_models
    #if (len(available_models.keys())) > 0:
    #    return
    # read available models
    with open('available_models.yml', 'r') as file:
        available_models = yaml.safe_load(file)
        return available_models

def update_model(model_id: str):
    with open('available_models.yml', 'r') as file:
        available_models = yaml.safe_load(file)
    global current_model_id
    global model
    if current_model_id != model_id:
        print("UPDATING MODEL ...")
        current_model_id = model_id
        current_model = available_models[current_model_id]
        print("CURRENT MODEL", current_model)
        # Reload model
        model = init_detector(
            current_model["config"],
            current_model["weights"],
            device=device
        )
    return model

app = FastAPI()


@app.get("/available_models")
def get_available_models():
    with open('available_models.yml', 'r') as file:
        available_models = yaml.safe_load(file)
    return list(available_models.keys())


def predict_images(filenames):
    global model
    results = inference_detector(
        model=model,
        imgs=filenames
    )

    bboxes = []
    for filename, result in zip(filenames, results):
        print(f"Processing file {filename} ...")
        for x in result[0]:
            if x[4] > 0.1:
                bboxes.append({
                    "filename": filename,
                    "class_index": 0,
                    "class_name": "spine",
                    "xmin": int(x[0]),
                    "ymin": int(x[1]),
                    "xmax": int(x[2]),
                    "ymax": int(x[3]),
                    "score": float(x[4])
                })
    return bboxes

@app.post("/image_inference/{model_id}")
async def upload_image(model_id: str, file: UploadFile = File(...)):
    model = update_model(model_id=model_id)

    # Specify the directory where you want to save the image
    save_directory = "data/tmp"

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save the uploaded image to the specified directory
    file_path = os.path.join(save_directory, file.filename)
    with open(file_path, "wb") as image_file:
        content = await file.read()
        image_file.write(content)

    print("Inference ...")
    time_start = time.time()
    predictions = predict_images([file_path])
    time_end = time.time()
    print("Inference took " + str(time_end-time_start) + " seconds")
    os.remove(file_path)

    return {
        "filename": file.filename,
        "saved_path": file_path,
        "bboxes": predictions,
    }
