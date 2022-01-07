import hydra
from omegaconf import OmegaConf, open_dict
from transformers import (
    AutoModelForMaskedLM,
    XLMRobertaConfig,
    BertConfig,
    DistilBertConfig,
    AutoConfig,
)


def initialize_model(cfg, teacher=None):
    """Teacher and student consists of tokenizer, model and embeddings.

    Returns:

    """
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        OmegaConf.update(cfg.model, "cfg.vocab_size", tokenizer.vocab_size)
    model, architecture_cfg = hydra.utils.instantiate(cfg.model, teacher=teacher)

    # If not max sequence length is given in tokenizer
    if tokenizer.model_max_length > int(1e20):
        tokenizer.model_max_length = architecture_cfg.max_position_embeddings

    if teacher is not None:
        # Delete old embedding layer to save some space
        model.base_model.embeddings = None
        # Only initialize multiple embeddings for students.
        embeddings = initialize_embeddings(cfg, teacher)
    else:
        # None for teacher
        embeddings = None

    return tokenizer, model, embeddings


def get_model(pretrained_model_name_or_path, use_pretrained_weights, cfg=None, **kwargs):
    use_automodel = False
    if 'distilbert-base-' in pretrained_model_name_or_path:
        config_class = DistilBertConfig
    elif 'bert-base-' in pretrained_model_name_or_path:
        config_class = BertConfig
    elif 'xlm-roberta-base' in pretrained_model_name_or_path:
        config_class = XLMRobertaConfig
    else:
        use_automodel = True
        # raise Exception('Add config_class corresponding to your model!')

    # Use from_pretrained to load the model with weights
    architecture_cfg = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                                  output_hidden_states=cfg.output_hidden_states,
                                                  output_attentions=cfg.output_attentions)
    if use_pretrained_weights:
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path,
                                                     output_hidden_states=cfg.output_hidden_states,
                                                     output_attentions=cfg.output_attentions)

    # Load the model from standard configuration without loading the weights
    else:
        if not use_automodel:
            architecture_cfg = config_class(**cfg)
        model = AutoModelForMaskedLM.from_config(architecture_cfg)

    return model, architecture_cfg


def initialize_embeddings(cfg, teacher=None):
    embeddings = {}
    for language in cfg.languages:
        model, _ = hydra.utils.instantiate(cfg.model, teacher=teacher)
        if model.__class__.__name__ == 'TinyModel':
            embeddings[language] = model.base.base_model.embeddings
        else:
            embeddings[language] = model.base_model.embeddings
    return embeddings
