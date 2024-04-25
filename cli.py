import click
import os
from tnc_process import Process
from detect import detect as _detect
from eval import evaluate

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config_dir', type=str, required=True, help='Directory containing the configuration files')
@click.option('--train_annotations', type=click.Path(exists=True, file_okay=True, dir_okay=False), default="data/annotations/instances_train2017.json", required=False, help='Path to the train annotations')
@click.option('--train_images', type=click.Path(exists=True, file_okay=False, dir_okay=True), default="data/train2017", required=False, help='Path to the train images')
@click.option('--val_annotations', type=click.Path(exists=True, file_okay=True, dir_okay=False), default="data/annotations/instances_val2017.json", required=False, help='Path to the validation annotations')
@click.option('--val_images', type=click.Path(exists=True, file_okay=False, dir_okay=True), default="data/val2017", required=False, help='Path to the validation images')
@click.option('--test_annotations', type=click.Path(exists=True, file_okay=True, dir_okay=False), default="data/annotations/instances_test2017.json", required=False, help='Path to the test annotations')
@click.option('--test_images', type=click.Path(exists=True, file_okay=False, dir_okay=True), default="data/test2017", required=False, help='Path to the test images')
@click.option('--model_type', type=str, required=True, help='Type of model to train')
@click.option('--model_name', type=str, required=True, help='Name of the model')
@click.option('--epochs', type=int, required=True, help='Number of training epochs (greater than 0)')
@click.option('--classes', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Classes file')
@click.option('--batch_size', type=int, default=2, help='Batch size for training (greater than 0, default: 2)')
@click.option('--work_dir', type=str, required=True, default=None, help='Working directory (default: None)')
def train(config_dir, train_annotations, train_images, val_annotations, val_images, test_annotations, test_images, model_type, model_name, epochs, classes, batch_size, work_dir):
    if not work_dir:
        work_dir = f"./runs/{model_type}/{model_name}"
    env = {
        "BATCH_SIZE": str(batch_size),
        "CLASSES_FILE": classes,
        "DATASET_TRAIN_ANNOTATION": train_annotations,
        "DATASET_TRAIN_IMAGES":  train_images,
        "DATASET_VAL_ANNOTATION": val_annotations,
        "DATASET_VAL_IMAGES": val_images,
        "DATASET_TEST_ANNOTATION": test_annotations,
        "DATASET_TEST_IMAGES": test_images,
        "EPOCHS": str(epochs),
    }
    p = Process(
        f"python train.py {config_dir}/{model_type}/{model_name}.py --work-dir {work_dir} --auto-resume",
        env=env,
    )
    p.run(stop_on_error=False)

@cli.command()
@click.option('--model_type', type=str, required=True, help='Type of model to use for detection')
@click.option('--model_name', type=str, required=True, help='Name of the model')
@click.option('--weight_file', type=str, required=True, help='Model weight file')
@click.option('--image_files', type=str, required=True, help='Glob path for images')
@click.option('--results_file', type=str, required=True, help='Name of the resulting CSV file')
@click.option('--batch_size', type=int, default=2, help='Batch size for training (greater than 0, default: 2)')
@click.option('--device', type=str, default='cuda:0', help='Device to use for detection (default: cuda:0)')
def detect(model_type, model_name, weight_file, image_files, results_file, batch_size, device):
    _detect(
        model_name=model_name,
        model_type=model_type,
        weight_file=weight_file,
        results_file=results_file,
        image_files=image_files.replace("'", ""),
        batch_size=batch_size,
        device=device
    )
    

@cli.command()
@click.option('--model_type', type=str, required=True, help='Type of model to use for detection')
@click.option('--model_name', type=str, required=True, help='Name of the model')
@click.option('--annotations', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Path to the annotations')
@click.option('--epochs', type=int, required=True, help='Number of training epochs (greater than 0)')
@click.option('--csv_file_pattern', type=str, required=True, help='Pattern for the CSV files ($i will be replaced by epoch number)')
@click.option('--results_file', type=str, required=True, help='Name of the resulting CSV file')
def eval(
    model_type: str,
    model_name: str,
    annotations: str,
    epochs: int,
    csv_file_pattern: str,
    results_file: str,
):
    evaluate(
        gt_file_path=annotations,
        model_type=model_type,
        model_name=model_name,
        csv_file_pattern=csv_file_pattern,
        results_file=results_file,
        max_epochs=epochs,
    )

if __name__ == '__main__':
    cli()