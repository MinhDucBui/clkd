from transformers import AutoModelForMaskedLM, AutoConfig
import hydra
from omegaconf import OmegaConf, open_dict
from transformers import (
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    DistilBertConfig,
    DistilBertForMaskedLM
)


def initialize_teacher_or_student(cfg):
    """Teacher and student consists of tokenizer, and model (currently).

    Returns:

    """
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        OmegaConf.update(cfg.model, "cfg.vocab_size", tokenizer.vocab_size)

    model, architecture_cfg = hydra.utils.instantiate(cfg.model)
    # If not max sequence length is given in tokenizer
    if tokenizer.model_max_length > int(1e20):
        tokenizer.model_max_length = architecture_cfg.max_position_embeddings

    return tokenizer, model


def get_automodel(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights:
        architecture_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, output_hidden_states=True), architecture_cfg
    else:
        architecture_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        architecture_cfg.vocab_size = cfg.vocab_size
        architecture_cfg.output_hidden_states = True
        return AutoModelForMaskedLM.from_config(architecture_cfg), architecture_cfg


def get_bert(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = BertConfig.from_pretrained(pretrained_model_name_or_path)
        return BertForMaskedLM.from_pretrained(pretrained_model_name_or_path, output_hidden_states=True), architecture_cfg
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = BertConfig.from_pretrained(pretrained_model_name_or_path)
        architecture_cfg.output_hidden_states = True
        return BertForMaskedLM(architecture_cfg), architecture_cfg
    else:
        architecture_cfg = BertConfig(**cfg)
        architecture_cfg.output_hidden_states = True
        return BertForMaskedLM(architecture_cfg), architecture_cfg


def get_distillbert(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = DistilBertConfig.from_pretrained(pretrained_model_name_or_path)
        return DistilBertForMaskedLM.from_pretrained(pretrained_model_name_or_path, output_hidden_states=True), architecture_cfg
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = DistilBertConfig.from_pretrained(pretrained_model_name_or_path)
        architecture_cfg.output_hidden_states = True
        return DistilBertForMaskedLM(architecture_cfg), architecture_cfg
    else:
        architecture_cfg = DistilBertConfig(**cfg)
        architecture_cfg.output_hidden_states = True
        return DistilBertForMaskedLM(architecture_cfg), architecture_cfg


def get_xlmr(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = XLMRobertaConfig.from_pretrained(pretrained_model_name_or_path)
        return XLMRobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path, output_hidden_states=True), architecture_cfg
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = XLMRobertaConfig.from_pretrained(pretrained_model_name_or_path)
        architecture_cfg.output_hidden_states = True
        return XLMRobertaForMaskedLM(architecture_cfg), architecture_cfg
    else:
        architecture_cfg = XLMRobertaConfig(**cfg)
        architecture_cfg.output_hidden_states = True
        return XLMRobertaForMaskedLM(architecture_cfg), architecture_cfg
