from transformers import AutoModelForMaskedLM, AutoConfig
from transformers import (
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    DistilBertConfig,
    DistilBertForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM,
)


def get_automodel(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights:
        return AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    else:
        architecture_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return AutoModelForMaskedLM.from_config(architecture_cfg)


def get_bert(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        return BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = BertConfig.from_pretrained(pretrained_model_name_or_path)
        return BertForMaskedLM.from_config(architecture_cfg)
    else:
        architecture_config = BertConfig(**cfg)
        return BertForMaskedLM(architecture_config)


def get_distillbert(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        return DistilBertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = DistilBertConfig.from_pretrained(pretrained_model_name_or_path)
        return DistilBertForMaskedLM(architecture_cfg)
    else:
        architecture_config = DistilBertConfig(**cfg)
        return DistilBertForMaskedLM(architecture_config)


def get_xlmr(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        return XLMRobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = XLMRobertaConfig.from_pretrained(pretrained_model_name_or_path)
        return XLMRobertaForMaskedLM(architecture_cfg)
    else:
        architecture_config = XLMRobertaConfig(**cfg)
        return XLMRobertaForMaskedLM(architecture_config)
