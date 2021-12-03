import hydra
from omegaconf import DictConfig


def config_callback(cfg: DictConfig, cb_cfg: DictConfig) -> DictConfig:
    """Amends configuration with user callback by configuration key.

    Hydra excels at depth-first, bottom-up config resolution. However,
    such a paradigm does not always allow you to elegantly express scenarios
    that are very relevant in experimentation. One instance, where :obj:`trident`
    levers :obj:`config_callback` is the `Huggingface datasets <https://huggingface.co/docs/datasets/>`_ integration.

    An example configuration may look like follows:

    .. code-block:: yaml

        config: # global config
          datamodule:
            dataset_cfg:
              # ${SHARED}
              _target_: datasets.load.load_dataset
              #     trident-integration into huggingface datasets
              #     to lever dataset methods within yaml configuration
              _method_:
                function:
                  _target_: src.utils.hydra.partial
                  _partial_: src.datamodules.utils.preprocessing.text_classification
                  tokenizer:
                    _target_: src.datamodules.utils.tokenization.HydraTokenizer
                    pretrained_model_name_or_path: roberta-base
                    max_length: 53
                batched: false
                num_proc: 12

              path: glue
              name: mnli

              # ${INDIVIDUAL}
              train:
                split: "train"
                # ${SHARED} will be merged into {train, val test} with priority for existing config
              val:
                split: "validation_mismatched+validation_matched"
              test:
                path: xtreme # overrides shared glue
                name: xnli # overrides shared mnli
                lang: de
                split: "test"

    Args:
        cfg:
        cb_cfg:

    Returns:

    """
    for key in cb_cfg:
        cfg = hydra.utils.call(cb_cfg.get(key), cfg=cfg, cfg_key=key)
    return cfg