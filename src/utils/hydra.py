import functools
from dataclasses import dataclass
from typing import Any, Callable, List
from hydra.utils import get_method
from src.models.modules.pooling import cls
from omegaconf import DictConfig, OmegaConf, open_dict
from typing import Union
import re


def expand(
    cfg: DictConfig,
    cfg_key: str,
    keys: Union[str, list[str]] = ["train", "val", "test"],
    gen_keys: bool = False,
) -> DictConfig:
    """Expands partial configuration of `keys` in `cfg` with the residual configuration.

    Most useful when configuring modules that have a substantial shared component.

    Applied by default on :obj:`dataset_cfg` (with :code:`create_keys=False`) and :obj:`dataloader_cfg` (with :code:`create_keys=True`) of :obj:`DataModule` config.

    Notes:
        - Shared config reflects all configuration excluding set :obj:`keys`.

    Args:
        keys (:obj:`Union[str, list[str])`):
            Keys that comprise dedicated configuration for which shared config will be merged.

        gen_keys (:obj:`bool`):
            Whether (:code:`True`) or not (:code:`False) to create :code:`keys` in :code:`cfg: with shared configuration if :code:`keys` do not exist yet.

    Example:
        :code:`expand(cfg, keys=["train", "val", "test"], create_keys=True)` with the following config

        .. code-block:: yaml

            dataloader_cfg:
                batch_size: 4
                num_workers: 8
                train:
                    batch_size: 8
                    shuffle: True
                test:
                    shuffle: False

        resolves to

        .. code-block:: yaml

            dataloader_cfg:
                train:
                    shuffle: True
                    batch_size: 8
                    num_workers: 8
                val:
                    batch_size: 4
                    num_workers: 8
                test:
                    shuffle: False
                    batch_size: 4
                    num_workers: 8

        while only the original config is the one being logged.
    """

    if isinstance(keys, str):
        keys = [keys]
    # Support Regex Strings
    keys_regex = [re.compile(key) for key in keys]
    shared_keys = [key for key in OmegaConf.select(cfg, cfg_key).keys() if not any(compiled_reg.match(key) for compiled_reg in keys_regex)]
    keys = [key for key in OmegaConf.select(cfg, cfg_key).keys() if any(compiled_reg.match(key) for compiled_reg in keys_regex)]
    # shared_keys = [key for key in cfg.keys() if key not in keys]
    cfg_excl_keys = OmegaConf.masked_copy(OmegaConf.select(cfg, cfg_key), shared_keys)
    for key in keys:
        if key in OmegaConf.select(cfg, cfg_key):
            # right-most gets priority
            OmegaConf.select(cfg, cfg_key)[key] = OmegaConf.merge(cfg_excl_keys, OmegaConf.select(cfg, cfg_key)[key])
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        for key in shared_keys:
            OmegaConf.select(cfg, cfg_key).pop(key)
    return cfg


@dataclass
class PartialWrapper:
    methods: List[Callable]

    def __call__(self, inputs) -> Any:
        for method in self.methods:
            inputs = method(inputs)
        return inputs


def partial(_partial_, *args, **kwargs):
    if isinstance(_partial_, list):
        methods = PartialWrapper([get_method(p) for p in _partial_])
        return methods
    return functools.partial(get_method(_partial_), *args, **kwargs)


def get_cls(outputs, batch):
    outputs["cls"] = cls(outputs.hidden_states[-1], batch["attention_mask"])
    return outputs


# list of dictionaries flattend to one dictionary
def prepare_retrieval_eval(outputs):
    num = outputs["cls"].shape[0]
    outputs["cls"] /= outputs["cls"].norm(2, dim=-1, keepdim=True)
    src_embeds = outputs["cls"][: num // 2]
    trg_embeds = outputs["cls"][num // 2:]
    # (1000, 1000)
    preds = src_embeds @ trg_embeds.T
    # targets = (
    #     torch.zeros((num // 2, num // 2)).fill_diagonal_(1).long().to(src_embeds.device)
    # )
    return {
        "preds": preds,
        # "targets": targets,
    }
