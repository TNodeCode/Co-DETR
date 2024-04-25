import os
import glob
import time
import numpy as np
import pandas as pd
from mmdet.apis import inference_detector, init_detector


def detect(
        model_type: str,
        model_name: str,
        weight_file: str,
        results_file: str,
        image_files: str,
        batch_size: int = 8,
        device: str = "cuda:0",
):
    run_dir = f"./runs/{model_type}/{model_name}"
    config_file_path = f"{run_dir}/{model_name}.py"
    weight_file_path = f"{run_dir}/{weight_file}"

    print("Config file", config_file_path)
    print("Weight file", weight_file_path)
    print("Device", device)
    print("Batch Size", batch_size)

    print("Loading model ...")
    model = init_detector(
        config_file_path,
        weight_file_path,
        device=device
    )
    print("Model loaded")

    filenames = glob.glob(image_files)
    n_files = len(filenames)
    n_batches=(n_files // batch_size) + 1

    bboxes = []
    durations = []
    for b in range((n_files // batch_size) + 1):
        if (len(filenames[b*batch_size:(b+1)*batch_size])) < 1:
            continue
        print(f"Processing batch {b}/{n_batches} ...")
        start = time.time()
        results = inference_detector(
            model=model,
            imgs=filenames[b*batch_size:(b+1)*batch_size]
        )
        end = time.time()
        durations.append(end - start)
        print("Finished batch in", end - start, "seconds")
        # Iterate over image results
        for i, result in enumerate(results):
            filename = os.path.basename(filenames[b*batch_size+i])
            # Iterate over detected bounding boxes
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

    csv_filename = f"{run_dir}/{results_file}"
    pd.DataFrame(bboxes).to_csv(csv_filename, index=False)
    print("Saved CSV file at", csv_filename)

    durations = np.array(durations)
    print("Inference took", durations.mean(), "per batch on average, std=", durations.std())
    print("Inference took", durations.mean() / batch_size, "per image on average, std=", durations.std() / batch_size)