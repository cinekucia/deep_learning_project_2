from lightning.pytorch.loggers import CSVLogger
from .session_preparation import prepare_session
from dataset.training_dataset import SpeechDataset

session_id_train = 0  

def train(config: dict, audio_dir: str):
    global session_id_train
    session_id_train += 1
    logger_name = f"train_logs_session_{session_id_train}"
    csv_logger = CSVLogger("logs", name=logger_name)

    trainer, pl_model, data = prepare_session(config, audio_dir, csv_logger, SpeechDataset)
    data.setup()
    trainer.fit(pl_model, data)
    pl_model.load_best_model()
    trainer.validate(pl_model, data)
    trainer.test(pl_model, data)