import os
import time
import torch
from mmdet.apis import inference_detector, init_detector
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

available_models = {
    "co_detr": {
        "config": "./projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo_spine.py",
        "weights": "./runs/co_dino_5scale_swin_large_16e_o365tococo_spine/epoch_25.pth",
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
current_model_id = "cascade_rcnn_run1"
current_model = init_detector(
    "./projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo_spine.py",
    "./runs/co_dino_5scale_swin_large_16e_o365tococo_spine/epoch_25.pth",
    device=device
)

def update_model(model_id: str):
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



@app.get("/available_models")
async def get_available_models():
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
