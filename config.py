import os

def get_classes(filename: str = None) -> list[str]:
    """
    Get array of class names from text file

    Parameters:
        filename: File where are class names a stored. File has one class name per line.

    Return:
        List of class names
    """
    if filename is None:
        filename = "classes.txt" if not os.getenv("CLASSES_FILE") else os.getenv("CLASSES_FILE")
    with open(filename, "r") as f:
        return f.read().strip().split("\n")


def get_train_annotation_file(default: str = "/data/annotations/instances_train2017.json") -> str:
    """
    Get train annotation file path.

    Parameters:
        default: Default value.

    Return:
        Dataset train annotation file path
    """
    return default if not os.getenv("DATASET_TRAIN_ANNOTATION") else os.getenv("DATASET_TRAIN_ANNOTATION")


def get_train_image_dir(default: str = "/data/train2017") -> str:
    """
    Get train images directory.

    Parameters:
        default: Default value.

    Return:
        Dataset train images directory
    """
    return default if not os.getenv("DATASET_TRAIN_IMAGES") else os.getenv("DATASET_TRAIN_IMAGES")


def get_val_annotation_file(default: str = "/data/annotations/instances_val2017.json") -> str:
    """
    Get validation annotation file path.

    Parameters:
        default: Default value.

    Return:
        Dataset validation annotation file path
    """
    return default if not os.getenv("DATASET_VAL_ANNOTATION") else os.getenv("DATASET_VAL_ANNOTATION")


def get_val_image_dir(default: str = "/data/val2017") -> str:
    """
    Get validation images directory.

    Parameters:
        default: Default value.

    Return:
        Dataset validation images directory
    """
    return default if not os.getenv("DATASET_VAL_IMAGES") else os.getenv("DATASET_VAL_IMAGES")


def get_test_annotation_file(default: str = "/data/annotations/instances_test2017.json") -> str:
    """
    Get test annotation file path.

    Parameters:
        default: Default value.

    Return:
        Dataset test annotation file path
    """
    return default if not os.getenv("DATASET_TEST_ANNOTATION") else os.getenv("DATASET_TEST_ANNOTATION")


def get_test_image_dir(default: str = "/data/test2017") -> str:
    """
    Get test images directory.

    Parameters:
        default: Default value.

    Return:
        Dataset test images directory
    """
    return default if not os.getenv("DATASET_TEST_IMAGES") else os.getenv("DATASET_TEST_IMAGES")
    
    
def get_batch_size(default: int = 4) -> int:
    """
    Get batch size

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Batch size that is used for training
    """
    return default if not os.getenv("BATCH_SIZE") else int(os.getenv("BATCH_SIZE"))
    
    
def get_number_of_epochs(default: int = 12) -> int:
    """
    Get number of epochs

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Number of epochs that the model will be trained
    """
    return default if not os.getenv("EPOCHS") else int(os.getenv("EPOCHS"))
    
    
def get_workers_per_gpu(default: int = 4) -> int:
    """
    Get workers per GPU

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Workers per GPU
    """
    return default if not os.getenv("WORKERS") else int(os.getenv("WORKERS"))