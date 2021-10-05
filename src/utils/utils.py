import logging
import os
import sys
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_subset_from_batch(batch, idx):
    new_batch = {}
    for key, value in batch.items():
        new_batch[key] = value[idx]

    return new_batch


def get_languages_from_mapping(language_mapping):
    languages = []
    potential_languages = list(language_mapping["lang_id"].keys())
    for single_language in potential_languages:
        single_language.split("_")


def create_language_mapping(distillation_type, s_lang, t_lang):
    """Create a mapping from language to a unique index. Furthermore, each source language has an additional index=0,
    and each target language has an additional index=1. E.g. "english": [0, 0] <-> english has (unique) index 0 and is
    a source language.

    Args:
        s_lang ():
        t_lang ():

    Returns:

    """
    mapping = {"lang_id": None, "id_lang": None}
    if distillation_type == "monolingual":
        mapping = monolingual_mapping(s_lang, t_lang)
    elif distillation_type == "bilingual":
        mapping = bilingual_mapping(s_lang, t_lang)

    return mapping


def bilingual_mapping(s_lang, t_lang):
    mapping = monolingual_mapping(s_lang, t_lang)
    index = 0
    if isinstance(t_lang, str):
        mapping["id_model"], _ = bilingual_parse_mapping_language(s_lang, t_lang, index=index)
    elif isinstance(t_lang, bool):
        sys.exit("No Target Language is given.")
    else:
        for single_target_language in t_lang:
            mapping["id_model"], index = bilingual_parse_mapping_language(s_lang, single_target_language, index=index)

    mapping_lang_id = {v[0]: [k] for k, v in mapping["id_model"].items()}
    mapping["model_id"] = mapping_lang_id
    return mapping


def monolingual_mapping(s_lang, t_lang):
    mapping = {}
    s_mapping_id_lang, s_index = monolingual_parse_mapping_language(s_lang, prefix=0, index=0)
    t_mapping_id_lang, _ = monolingual_parse_mapping_language(t_lang, prefix=1, index=s_index)
    s_mapping_id_lang.update(t_mapping_id_lang)
    mapping["id_lang"] = s_mapping_id_lang
    mapping_lang_id = {v[0]: [k, v[1]] for k, v in mapping["id_lang"].items()}
    mapping["lang_id"] = mapping_lang_id
    mapping["model_id"] = mapping["lang_id"]
    mapping["id_lang"] = mapping["id_lang"]
    return mapping


def bilingual_parse_mapping_language(s_lang, language, index: int):
    mapping_id_lang = {}
    if not index:
        index = 0
    if isinstance(s_lang, str):
        mapping_id_lang[index] = [s_lang + "_" + language]
    elif isinstance(s_lang, bool):
        pass
    else:
        for single_language in s_lang:
            mapping_id_lang[index] = [single_language + "_" + language]
            index += 1

    return mapping_id_lang, index


def monolingual_parse_mapping_language(language, prefix: int, index: int):
    mapping_id_lang = {}
    if not index:
        index = 0
    if isinstance(language, str):
        mapping_id_lang[index] = [language, prefix]
    elif isinstance(language, bool):
        pass
    else:
        for single_language in language:
            mapping_id_lang[index] = [single_language, prefix]
            index += 1

    return mapping_id_lang, index


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.train.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.train.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "train",
                "distillation",
                "teacher",
                "student",
                "datamodule",
                "callbacks",
                "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config.train.trainer
    hparams["student"] = config["student"]
    hparams["teacher"] = config["teacher"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    # hparams["student/params_total"] = sum(p.numel() for p in model.parameters())
    # hparams["student/params_trainable"] = sum(
    #    p.numel() for p in model.parameters() if p.requires_grad
    # )
    # hparams["student/params_not_trainable"] = sum(
    #    p.numel() for p in model.parameters() if not p.requires_grad
    # )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
