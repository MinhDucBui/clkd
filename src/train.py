from typing import List, Optional
from src.distillation.base_module import BaseModule
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
    seed_everything,
)
from src.callbacks.config_callback import config_callback
from pytorch_lightning.loggers import LightningLoggerBase
from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for cb_name, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
            elif cb_name == "config_callback":
                log.info(f"Instantiating config callback for <{cb_name}>")
                config = config_callback(config, cb_conf)

    # Init Module
    distillation: LightningModule = BaseModule(cfg=config)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=distillation,
        datamodule=distillation.datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=distillation,
                datamodule=distillation.datamodule
                )

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.train.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=distillation,
        datamodule=distillation.datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
