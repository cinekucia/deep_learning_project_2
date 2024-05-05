from enum import Enum

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

class SplitType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
