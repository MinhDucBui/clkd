from src.utils.utils import convert_cfg_tuple
from transformers import BertForMaskedLM
import copy
import sys


def initialize_model_different_embeddings(model, model_name, languages):
    new_models = {}
    for language in languages:
        new_models[language] = copy.deepcopy(model)
        if isinstance(model, BertForMaskedLM):
            new_models[language].base_model.encoder = model.base_model.encoder
            new_models[language].cls = model.cls
        else:
            sys.exit("Get Layer for this Model Type is not implemented!!")
    return new_models


def embedding_sharing(embedding_sharing_cfg):
    if embedding_sharing_cfg == "in_each_model":
        pass
    else:
        new_cfg = []
        for embedding_sharing_tuple in embedding_sharing_cfg:
            new_cfg.append(convert_cfg_tuple(embedding_sharing_tuple))


def weight_sharing(weight_sharing_cfg, students, student_mapping):
    if weight_sharing_cfg == "in_each_model":
        pass
    else:
        new_cfg = []
        for weight_sharing_tuple in weight_sharing_cfg:
            new_cfg.append(convert_cfg_tuple(weight_sharing_tuple))

        for sharing_tuple in new_cfg:
            origin_id = get_id(sharing_tuple[0], student_mapping)
            for single in sharing_tuple[1:]:
                replace_student_id = get_id(single, student_mapping)
                replace_layer(students,
                              origin_id=origin_id, origin_layer_n=int(sharing_tuple[0][1]),
                              replace_student_id=replace_student_id, replace_layer_n=int(single[1]))


def get_id(sharing_tuple, student_mapping):
    return int(student_mapping["model_id"][sharing_tuple[0]]["idx"])


def replace_layer(students, origin_id, origin_layer_n, replace_student_id, replace_layer_n):
    if isinstance(students[replace_student_id], BertForMaskedLM):
        students[replace_student_id].base_model.encoder.layer[replace_layer_n] = \
            students[origin_id].base_model.encoder.layer[origin_layer_n]

        # Debug:
        # test_lang = list(students[replace_student_id].keys())[1]
        # print(students[replace_student_id][test_lang].base_model.encoder)
        # print(students[replace_student_id][test_lang].base_model.encoder.layer)
    else:
        sys.exit("Get Layer for this Model Type is not implemented!!")