import hydra
import torch.nn as nn
import copy
from omegaconf import OmegaConf, open_dict
from transformers import (
    AutoModelForMaskedLM,
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    DistilBertConfig,
    DistilBertForMaskedLM,
    AutoConfig,
)


def change_embedding_layer(model, model_idx, embeddings, language):
    if model.__class__.__name__ == 'TinyModel':
        model.base.base_model.embeddings = embeddings[model_idx][language]
    else:
        model.base_model.embeddings = embeddings[model_idx][language]


def initialize_embeddings(cfg, teacher=None):
    embeddings = {}
    for language in cfg.languages:
        model, _ = hydra.utils.instantiate(cfg.model, teacher=teacher)
        # if torch.cuda.is_available():
        #    model.base_model.embeddings = model.base_model.embeddings.to(device='cuda')
        if model.__class__.__name__ == 'TinyModel':
            embeddings[language] = model.base.base_model.embeddings
        else:
            embeddings[language] = model.base_model.embeddings
    return embeddings


def initialize_teacher_or_student(cfg, teacher=None):
    """Teacher and student consists of tokenizer, and model (currently).

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

    return tokenizer, model


def get_tiny_model(pretrained_model_name_or_path, teacher, mapping, weights_from_teacher=None,
                   cfg=None, **kwargs):

    if 'distilbert' in pretrained_model_name_or_path:
        model_class, config_class = DistilBertForMaskedLM, DistilBertConfig
    elif 'xlm' in pretrained_model_name_or_path:
        model_class, config_class = XLMRobertaForMaskedLM, XLMRobertaConfig
    elif 'bert' in pretrained_model_name_or_path:
        model_class, config_class = BertForMaskedLM, BertConfig
    else:
        model_class, config_class = AutoModelForMaskedLM, AutoConfig

    architecture_cfg = config_class(**cfg)

    class TinyModel(nn.Module):
        def __init__(self, config, teacher_model, init_weights, mapp, fit_size=768):
            super(TinyModel, self).__init__()
            self.student_hidden_size = config.hidden_size
            self.init_weights = init_weights
            self.fit_size = fit_size
            self.mapping = mapp
            self.teacher_model = teacher_model

            self.base = AutoModelForMaskedLM.from_config(config)
            # It's possible to init student's weights from teacher only if both agree on hidden dimensionality
            if self.fit_size == self.student_hidden_size:
                self.init_weights_from_teacher()

            self.projections = nn.ModuleList(
                [nn.Linear(cfg.hidden_size, self.fit_size) for _ in range(config.num_hidden_layers + 1)])
            
            self.base_model = self.base.base_model

        def init_weights_from_teacher(self):
            """
            The function works assuming that name of the module consists of following pattern-->"*.layer.no_of_layer.*"
            """

            # Copy teacher weights from corresponding transformer lays
            if self.init_weights.transformer_blocks:
                new_params = {}
                for key, value in self.mapping.items():
                    for name_t, param_t in self.teacher_model.named_parameters():
                        layer = ''.join(['layer.', str(value), '.'])
                        if layer in name_t:
                            name_s = name_t.replace(layer, ''.join(['layer.', str(key), '.']))
                            new_params[name_s] = copy.deepcopy(param_t)
                for name_s, param_t in self.base.named_parameters():
                    if name_s in new_params:
                        #print("Initialize {} from teachers weight".format(name_s))
                        param_t = new_params[name_s]

            # Copy teacher weights from embedding layer
            if self.init_weights.embeddings:
                new_emb_params = {}
                for name_t, param_t in self.teacher_model.named_parameters():
                    if 'embeddings' in name_t:
                        new_emb_params[name_t] = copy.deepcopy(param_t)
                
                for name_s, param_s in self.base.named_parameters():
                    for key, value in new_emb_params.items():
                        if name_s == key:
                            #print("Initialize {} from teachers weight".format(name_s))
                            param_s = value

        def forward(self, input_ids, token_type_ids=None,
                    attention_mask=None, labels=None):

            outputs = self.base(input_ids, token_type_ids, attention_mask)

            if self.student_hidden_size != self.fit_size:
                outputs['hidden_states'] = [self.projections[i](outputs["hidden_states"][i]) for i in
                                            range(len(self.projections))]

            return outputs

    return TinyModel(architecture_cfg, teacher, weights_from_teacher, mapping), cfg


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
