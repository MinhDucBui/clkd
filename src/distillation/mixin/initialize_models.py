from src.models.model import initialize_model
import hydra
from src.utils.utils import initialize_evaluation_cfg
from src.utils.mappings import create_mapping
from src.utils import utils
from src.utils.utils import convert_cfg_tuple
import torch
from transformers import BertForMaskedLM
import sys

LOOKUP_TABLE = {BertForMaskedLM: {"layer": "base_model.encoder.layer"}}
log = utils.get_logger(__name__)


class InitializeModelsMixin:
    r"""Mixin for base model to define evaluation loop largely via hydra.

    """

    def __init__(self) -> None:
        super().__init__()
        self.teacher_cfg = self.cfg.teacher
        self.students_cfg = self.cfg.students
        self.individual_students_cfg = self.cfg.students.individual
        self.students_model_cfg = {}
        for model_name, model_cfg in self.individual_students_cfg.items():
            if "student_" not in model_name:
                continue
            self.students_model_cfg[model_name] = model_cfg
        self.embedding_sharing_cfg = self.cfg.students.embed_sharing
        self.weight_sharing_cfg = self.cfg.students.weight_sharing_across_students

        self.evaluation_cfg = initialize_evaluation_cfg(self.cfg.evaluation)

        # Map language to id, student to languages and get validation tasks
        self.language_mapping, self.student_mapping, self.validation_mapping, self.val_dataset_mapping \
            = create_mapping(self.students_cfg, self.evaluation_cfg, self.cfg.datamodule)

        self.number_of_models = len(self.student_mapping["model_id"])

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{self.teacher_cfg.model._target_}>")
        self.teacher_tokenizer, self._teacher, self.teacher_outputs = None, None, {}
        self.initialize_teacher()

        # Init Students
        self.model, self.student_tokenizers, self.loss, self.embeddings = [], [], [], []
        self.initialize_student_components()

    def initialize_teacher(self):
        self.teacher_tokenizer, self._teacher, _ = initialize_model(self.teacher_cfg)
        self._teacher.eval()

    def initialize_student_components(self):
        for model_name, model_cfg in self.students_model_cfg.items():
            tokenizer, model, embeddings = initialize_model(model_cfg, self._teacher)
            exec("self.%s = %s" % (model_name, "model"))
            for language, embedding in embeddings.items():
                exec("self.%s = %s" % (model_name + "_" + language, "embedding"))
            self.model.append(model)
            self.embeddings.append(embeddings)
            self.student_tokenizers.append(tokenizer)
            self.loss.append(hydra.utils.instantiate(model_cfg["loss"]))

        self.embedding_sharing(self.embeddings, self.embedding_sharing_cfg, self.student_mapping)
        self.weight_sharing(self.weight_sharing_cfg, self.model, self.student_mapping)
        self.tie_output_embeddings(self.students_cfg.tie_output_embeddings, self.model, self.embeddings)

    def tie_output_embeddings(self, tie_output_embeddings_cfg, models, embeddings):

        if not tie_output_embeddings_cfg:
            return

        for index in range(len(models)):
            assert len(embeddings[index]) == 1, "Tie Output Embedding only for monolingual implemented!"
            language = list(embeddings[index].keys())[0]
            input_embeddings = embeddings[index][language].word_embeddings
            model_type = models[index].base.config.model_type
            output_embeddings = None
            if "xlm-roberta" == model_type:
                output_embeddings = models[index].base.lm_head.decoder
            elif "bert" == model_type:
                output_embeddings = models[index].base.cls.predictions.decoder
            assert output_embeddings, "Model Type not implemented yet in tie_output_embeddings"
            output_embeddings.weight = input_embeddings.weight
            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = torch.nn.functional.pad(
                    output_embeddings.bias.data,
                    (
                        0,
                        output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings

    def embedding_sharing(self, embeddings, embedding_sharing_cfg, student_mapping):
        if embedding_sharing_cfg == "in_each_model":
            for model_embeddings in embeddings:
                origin_language = list(model_embeddings.keys())[0]
                origin_embedding = model_embeddings[origin_language]
                for language in model_embeddings.keys():
                    model_embeddings[language] = origin_embedding
        elif embedding_sharing_cfg == "in_overlapping_language":
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    overlapping_embeddings = list(embeddings[i].keys() & embeddings[j].keys())
                    for overlapping_lang in overlapping_embeddings:
                        origin_embedding = embeddings[i][overlapping_lang]
                        embeddings[j][overlapping_lang] = origin_embedding
        else:
            new_cfg = []
            for embedding_sharing_tuple in embedding_sharing_cfg:
                new_cfg.append(convert_cfg_tuple(embedding_sharing_tuple))
            for sharing_tuple in new_cfg:
                origin_id = get_id(sharing_tuple[0], student_mapping)
                origin_language = sharing_tuple[0][1]
                origin_embedding = embeddings[origin_id][origin_language]
                origin_name = self.student_mapping["id_model"][origin_id]["model_name"] + "_" + origin_language
                for single in sharing_tuple[1:]:
                    replace_student_id = get_id(single, student_mapping)
                    replace_language = single[1]
                    replace_name = self.student_mapping["id_model"][replace_student_id]["model_name"] + "_" + replace_language
                    embeddings[replace_student_id][replace_language] = origin_embedding
                    exec("self.%s = self.%s" % (replace_name, origin_name))

        # Debug:
        # debug_id_embeddings(embeddings)

    def weight_sharing(self, weight_sharing_cfg, students, student_mapping):
        if not weight_sharing_cfg or weight_sharing_cfg == "in_each_model":
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
    for key, value in LOOKUP_TABLE.items():
        if isinstance(students[replace_student_id], key):
            exec("students[" + str(replace_student_id) + "]." + value["layer"] + "[" + str(replace_layer_n) + "] = "
                 + "students[" + str(origin_id) + "]." + value["layer"] + "[" + str(origin_layer_n) + "]")

            # Debug:
            # test_lang = list(students[replace_student_id].keys())[1]
            # print(students[replace_student_id][test_lang].base_model.encoder)
            # print(students[replace_student_id][test_lang].base_model.encoder.layer)
        else:
            sys.exit("Get Layer for this Model Type is not implemented!!")
