from lightning.pytorch.loggers import CSVLogger
from .session_preparation import prepare_session
from settings import ALL_CLASSES
from dataset.training_dataset import KaggleTestDataset

session_id_test = 0

def test(config: dict, audio_dir: str, model_checkpoint: str) -> tuple[list[str], list[str]]:
    global session_id_test
    session_id_test += 1  
    
    logger_name = f"test_logs_session_{session_id_test}"
    csv_logger = CSVLogger("logs", name=logger_name)
    
    trainer, pl_model, data = prepare_session(config, audio_dir, csv_logger, KaggleTestDataset)
    data.setup("test")
    pl_model.load_local(model_checkpoint)
    pl_model.log_test = False
    trainer.test(pl_model, data)

    return (
        data.file_names,
        [ALL_CLASSES[class_id] for class_id in pl_model.test_class_ids.numpy().tolist()],
    )
