from transformers import AutoModelForMaskedLM, AutoConfig
from transformers import (
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    DistilBertConfig,
    DistilBertForMaskedLM
)


def get_automodel(pretrained_model_name_or_path, use_pretrained_weights, output_hidden_states, output_attentions,
                  cfg=None, **kwargs):

    if use_pretrained_weights:
        return AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path,
                                                    output_hidden_states=output_hidden_states,
                                                    output_attentions=output_attentions,
                                                    **kwargs
                                                    )
    else:
        architecture_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                                      output_hidden_states=output_hidden_states,
                                                      output_attentions=output_attentions,
                                                      **kwargs
                                                      )
        architecture_cfg.vocab_size = cfg.vocab_size
        return AutoModelForMaskedLM.from_config(architecture_cfg)


def get_bert(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        return BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = BertConfig.from_pretrained(pretrained_model_name_or_path)
        return BertForMaskedLM(architecture_cfg)
    else:
        architecture_cfg = BertConfig(**cfg)
        return BertForMaskedLM(architecture_cfg)


def get_distillbert(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        return DistilBertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = DistilBertConfig.from_pretrained(pretrained_model_name_or_path)
        return DistilBertForMaskedLM(architecture_cfg)
    else:
        architecture_cfg = DistilBertConfig(**cfg)
        return DistilBertForMaskedLM(architecture_cfg)


def get_xlmr(pretrained_model_name_or_path, use_pretrained_weights, cfg=None):
    if use_pretrained_weights and pretrained_model_name_or_path:
        return XLMRobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    elif not use_pretrained_weights and pretrained_model_name_or_path:
        architecture_cfg = XLMRobertaConfig.from_pretrained(pretrained_model_name_or_path)
        return XLMRobertaForMaskedLM(architecture_cfg)
    else:
        architecture_cfg = XLMRobertaConfig(**cfg)
        return XLMRobertaForMaskedLM(architecture_cfg)
