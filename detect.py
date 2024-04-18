import time
import pandas as pd
from mmdet.apis import inference_detector, init_detector

model = init_detector(
    "./projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo_spine.py",
    "./runs/co_dino_5scale_swin_large_16e_o365tococo_spine/epoch_25.pth",
    device="cpu"
)

filenames = [
    "C:/Users/tilof/PycharmProjects/DeepLearningProjects/MasterThesis/CVDataInspector/datasets/spine/train/aidv853_date220321_tp1_stack3_sub11_layer116.png",
]

start = time.time()
results = inference_detector(
    model=model,
    imgs=filenames
)
end = time.time()

print("TIME", end - start)
print("RESULTS", results, len(results))
print("RESULT", results[0])

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

pd.DataFrame(bboxes).to_csv("co_detr_results.csv")