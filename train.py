import warnings
warnings.simplefilter('ignore', UserWarning)

import hydra
from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.datasets.lightning_datasets import LightningEnhHFLibriDataset
from src.orchestrator import LightningVoiceFixer

def train_pipeline(cfg: DictConfig) -> None:

    # setup dataset
    data_module = LightningEnhHFLibriDataset(
        cfg.dataset, batch_size=cfg.run.batch_size,
        )
    data_module.prepare_data()

    #########
    # TRAIN #
    #########

    data_module.setup(stage='fit')

    # initialise orchestrator
    orchestrator = LightningVoiceFixer(cfg.model, cfg.optim, sample_rate=cfg.dataset.sample_rate)

    # initialize callbacks
    checkpoint = ModelCheckpoint(monitor='valid_loss', filename='best', save_top_k=1)
    logger = TensorBoardLogger(cfg.run.logs_dir)  # logs and checkpoints will be stored in logs_dir

    # initialize Trainer
    trainer = Trainer(
        gpus=cfg.run.num_gpus, 
        callbacks=[checkpoint,], 
        logger=logger, 
        max_epochs=cfg.run.epochs, 
        accumulate_grad_batches=cfg.run.accumulate_grad_batches,
        strategy=DDPPlugin(find_unused_parameters=False) if cfg.run.num_gpus > 1 else None)

    # train
    trainer.fit(orchestrator, datamodule=data_module)

@hydra.main(config_path='configs', config_name='main')
def main(cfg: DictConfig):
    train_pipeline(cfg)

if __name__ == '__main__':
    main()