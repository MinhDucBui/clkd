import logging
import warnings
from typing import List, Sequence
import torch
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
import os.path
import sys
import requests
import lzma, shutil
import copy


def append_torch_in_dict(dict_to_add, dict_to_extend):
    for key, value in dict_to_add.items():
        if key not in dict_to_extend.keys():
            dict_to_extend[key] = dict_to_add[key]
        else:
            dict_to_extend[key] = torch.cat((dict_to_extend[key], dict_to_add[key]), dim=0)
    return dict_to_extend


# TODO: Is redundant when we have new student cfg structure
def get_corresponding_language_pairs(language_mapping):
    """Format: [[l1, l2, ...], [l1], ...]

    Returns:

    """
    language_pairs = []
    for value in language_mapping["id_lang"].values():
        if value[1] == "src":
            language_pairs += [[value[0], value_2[0]] for value_2 in language_mapping["id_lang"].values() if
                               value_2[1] != "src"]
    return language_pairs


# TODO: Later we pass relationship etc. (more information) when student cfg available.
#  This function is badly written, but is only temporarily.
def logger_naming(eval_cfg, language_mapping, stage="val", *args, **kwargs):
    """Format: language_pair (distilltype)/task/metric/dataset/language, e.g. ss_ht (bilingual)/mlm/perplexity/ss.

    Args:
        language_mapping:
        *args:
        **kwargs:

    Returns:

    """
    language_pairs = get_corresponding_language_pairs(language_mapping)
    source_model = [value[0] for key, value in language_mapping["model_id"].items() if value[1] == "src"][0]
    logger_names = []
    # Model
    for model_idx in range(len(language_mapping["model_id"])):
        new_item = {}
        model_languages = get_model_language(model_idx, language_mapping)
        new_item["model"] = name_model_for_logger(model_languages)
        new_item["model_id"] = model_idx
        # Task
        for task, task_cfg in eval_cfg.items():
            new_item["task_name"] = task_cfg.logger.name
            # Metric
            for metric_name in getattr(task_cfg, "metrics").keys():
                new_item["metric_name"] = metric_name
                # Dataset
                for language_pair in language_pairs:
                    # If they share items
                    if (len(model_languages) == 1 and bool(set(language_pair) & set(model_languages))) \
                            or (len(model_languages) > 1 and set(language_pair).issubset(model_languages)):
                        # Get aggregration
                        agg_lang = task_cfg.logger.get("aggregate_language", "pair")
                        if agg_lang == "pair":
                            if len(model_languages) == 1 and language_mapping["lang_id"][model_languages[0]][1] == "src":
                                continue
                            new_item["dataset"] = "_".join(language_pair)
                            new_item["selected_language"] = "_".join(language_pair)
                            # TODO: Should be in new student cfg architecture
                            if len(model_languages) == 1:
                                new_item["eval_with"] = source_model
                            else:
                                new_item["eval_with"] = ""
                            logger_names.append(copy.deepcopy(new_item))

                        elif agg_lang == "single":
                            for single_lang in language_pair:
                                #if len(model_languages) == 1 and language_mapping["lang_id"][single_lang][1] == "src":
                                #    continue
                                if not bool(set([single_lang]) & set(model_languages)):
                                    continue
                                new_item["dataset"] = "_".join(language_pair)
                                new_item["selected_language"] = single_lang
                                new_item["eval_with"] = ""
                                logger_names.append(copy.deepcopy(new_item))

    for item in logger_names:
        item["logger_name"] = "/".join([item["model"], stage, item["task_name"], item["metric_name"],
                                        item["dataset"] + "(dataset)", item["selected_language"]])

    return logger_names


# TODO: What if two student have same languages? Not unique in logger. Not allow this situation?
def name_model_for_logger(languages):
    if len(languages) == 1:
        distilltype = "(monolingual)"
    elif len(languages) == 2:
        distilltype = "(bilingual)"
    else:
        distilltype = "(multilingual)"

    return "_".join(languages) + distilltype


def get_model_language(model_idx, language_mapping):
    return language_mapping["id_model"][model_idx][0].split("_")


def keep_only_model_forward_arguments(model, batch, remove_additional_keys=[]):
    cleaned_batch = {key: value for key, value in batch.items() if key not in remove_additional_keys}
    cleaned_batch = {key: value for (key, value) in cleaned_batch.items() if
                     key in model.forward.__code__.co_varnames}

    return cleaned_batch


# Utils for Data Module
def download_file(language, output_folder):
    """Download file from cc100.

    Args:
        language:
        output_folder:

    Returns:

    """
    file_name = '{language}.txt.xz'.format(language=language)
    link = 'http://data.statmt.org/cc-100/' + file_name
    output_file = os.path.join(output_folder, file_name)
    with open(output_file, "wb") as f:
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            chunk_size = int(total_length / 100)
            for data in response.iter_content(chunk_size):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s] %s%%" % ('=' * done, ' ' * (50 - done), done * 2))
                sys.stdout.flush()
            print("\n")


def decompress_xz(input_file):
    """Function to decompress .xz file (cc100 format).

    Args:
        input_file:

    Returns:

    """
    input_file = Path(input_file)
    destination_dir = os.path.dirname(input_file)
    with lzma.open(input_file) as compressed:
        output_path = Path(destination_dir) / input_file.stem
        with open(output_path, 'wb') as destination:
            try:
                shutil.copyfileobj(compressed, destination)
            except EOFError:
                sys.exit("File {} is corrupted. Please Delete the file.".format(input_file))


# Utils for Model
def get_language_subset_batch(batch, language_mapping, model_languages):
    """Each batch consists of samples from multiple languages, but each model corresponds to a subset of languages
    Idea: Get only the samples that corresponds to the model's languages and get row index of the corresponding
    samples in the batch.

    Args:
        batch:
        language_mapping:
        model_languages:

    Returns:

    """

    idx = get_language_subset_index(language_mapping, batch["language"], model_languages)
    # Get corresponding Batch
    subset_batch = get_subset_dict(batch, idx)

    return subset_batch, idx


def get_language_subset_index(language_mapping, batch_language, model_languages):
    """Get index of samples in batch that corresponds to the model's languages.

    Args:
        language_mapping:
        batch_language:
        model_languages:

    Returns:

    """
    idx = None
    for index, single_language in enumerate(model_languages):
        language_id = language_mapping["lang_id"][single_language][0]
        subset_index = (batch_language == torch.tensor(language_id)).nonzero()[:, 0]
        if index == 0:
            idx = subset_index
        else:
            idx = torch.cat((idx, subset_index), 0)
    return idx


def get_subset_dict(full_set: dict, idx: torch.Tensor):
    """Get subset of batches that are contained in a dictionary.

    Args:
        full_set:
        idx:

    Returns:

    """
    subset = {}
    for key, value in full_set.items():
        # TODO: Fix for now. Wait for Marlena
        if key == "hidden_states":
            continue
        subset[key] = value[idx]

    return subset


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


def monolingual_parse_mapping_language(language, prefix: str, index: int):
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


# Utils for Hydra
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
