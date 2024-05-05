from enum import Enum

ENTITY = "dl-mini"
PROJECT = "DL_PROJECT_RNN"
NAMED_CLASSES = []
UNKNOWN_CLASS = "unknown"
ALL_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", UNKNOWN_CLASS]
AUDIO_FILE_METADATA = {
    "sample_rate": 16000,
    "encoding": "PCM_S",
    "bits_per_sample": 16,
    "num_channels": 1,
}
NUM_CLASSES = len(ALL_CLASSES)
DATA_DIR = "../data"


class JobType(Enum):
    UPLOAD_DATA = "upload-data"
    UPLOAD_CONFIG = "upload-config"
    DOWNLOAD_DATA = "download-data"
    TRAINING = "training"
    KAGGLE_TEST = "kaggle-test"


class ArtifactType(Enum):
    DATASET = "dataset"
    MODEL = "model"
    CONFIG_FILE = "config-file"
    KAGGLE_RESULTS = "kaggle-results"


class SplitType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
