import torch.nn as nn
import copy
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig
)
from src.utils import utils
import re 

log = utils.get_logger(__name__)


class TinyModel(nn.Module):
    def __init__(self, config, teacher_model, init_weights, mapp):
        super(TinyModel, self).__init__()
        self.student_hidden_size = config.hidden_size
        self.init_weights = init_weights
        self.mapping = mapp

        self.base = AutoModelForMaskedLM.from_config(config)
        self.fit_size = teacher_model.config.hidden_size
        # It's possible to init student's weights from teacher only if both agree on hidden dimensionality
        if self.fit_size == self.student_hidden_size:
            self.init_weights_from_teacher(teacher_model)

        if self.student_hidden_size != self.fit_size:
            self.projections = nn.ModuleList(
                [nn.Linear(config.hidden_size, self.fit_size) for _ in range(config.num_hidden_layers + 1)])
            
        #self.base_model = self.base.base_model

    def init_weights_from_teacher(self, teacher_model):
        """
        The function works assuming that name of the module consists of following pattern-->"*.layer.no_of_layer.*"
        """

        # Copy teacher weights from corresponding transformer lays
        if self.init_weights.transformer_blocks:
            new_params = {}
            for key, value in self.mapping.items():
                for name_t, param_t in teacher_model.named_parameters():
                    layer = ''.join(['layer.', str(value), '.'])
                    if layer in name_t:
                        name_s = name_t.replace(layer, ''.join(['layer.', str(key), '.']))
                        new_params[name_s] = copy.deepcopy(param_t)
                        log.info("Initialize {} from teachers weight {}".format(name_s, name_t))
            for name_s, param_s in self.base.named_parameters():
                if name_s in new_params:
                    param_t = new_params[name_s]
                    exec_name_s = re.sub(r'.([0-9]+).' , r'[\1].', name_s)
                    exec("self.base.%s = new_params['%s']" % (exec_name_s, name_s))
                    #param_t = new_params[name_s]

        # Copy teacher weights from embedding layer
        if self.init_weights.embeddings:
            new_emb_params = {}
            for name_t, param_t in teacher_model.named_parameters():
                if 'embeddings' in name_t:
                    new_emb_params[name_t] = copy.deepcopy(param_t)

            for name_s, param_s in self.base.named_parameters():
                for key, value in new_emb_params.items():
                    if name_s == key:
                        log.info("Initialize {} from teachers embedding".format(name_s))
                        exec_name_s = re.sub(r'.([0-9]+).' , r'[\1].', name_s)
                        exec("self.base.%s = value" % (exec_name_s))

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):

        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,)

        if self.student_hidden_size != self.fit_size:
            outputs['hidden_states'] = [self.projections[i](outputs["hidden_states"][i]) for i in
                                        range(len(self.projections))]

        return outputs
