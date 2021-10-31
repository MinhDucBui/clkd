from omegaconf import OmegaConf
import copy
from src.utils.utils import name_model_for_logger


def create_mapping(students_cfg):
    language_mapping = create_language_mapping(students_cfg.individual)
    student_mapping = create_model_mapping(students_cfg.individual)
    validation_mapping = create_validation_mapping(students_cfg.evaluation, student_mapping, stage="val")
    return language_mapping, student_mapping, validation_mapping


def create_model_mapping(students_model_cfg):
    mapping = {"model_id": {}, "id_model": {}}
    index = 0
    for key in list(students_model_cfg.keys()):
        if "student_" not in key:
            continue
        mapping["model_id"][key] = {"idx": index, "languages": students_model_cfg[key]["languages"]}
        mapping["id_model"][index] = {"model_name": key, "languages": students_model_cfg[key]["languages"]}
        index += 1
    return OmegaConf.create(mapping)


def create_language_mapping(students_model_cfg):
    mapping = {"lang_id": {}, "id_lang": {}}
    languages = []
    for key in sorted(list(students_model_cfg.keys())):
        if "student_" not in key:
            continue
        languages += students_model_cfg[key]["languages"]

    for index, language in enumerate(set(languages)):
        mapping["lang_id"][language] = index
        mapping["id_lang"][index] = language

    return OmegaConf.create(mapping)


def create_validation_mapping(evaluation_cfg, model_mapping, stage="val"):
    """Format: language_pair (distill type)/task/metric/dataset/language, e.g. ss_ht (bilingual)/mlm/perplexity/ss.

    Args:
        stage:
        model_mapping:
        evaluation_cfg:

    Returns:

    """

    logger_names = []

    # Task
    for task, task_cfg in evaluation_cfg.items():
        new_item = {}
        for eval_model_tuple in task_cfg["evaluate_with"]:
            new_item["task_name"] = task_cfg.logger.name
            new_item["dataset"] = [eval_model[1] for eval_model in eval_model_tuple]
            for metric_name in getattr(task_cfg, "metrics").keys():
                new_item["metric_name"] = metric_name
                new_item["cfg"] = {key: task_cfg[key] for key in ["apply", "step_outputs", "metrics"]}
                new_item["model_language"] = []
                if getattr(task_cfg, "aggregate", True):
                    new_item["model_name"] = eval_model_tuple[0][0]
                    new_item["model_idx"] = model_mapping["model_id"][new_item["model_name"]]["idx"]
                    new_item["model_language"] = model_mapping["model_id"][new_item["model_name"]]["languages"]
                    if len(eval_model_tuple) > 1:
                        for model_tuple in eval_model_tuple:
                            model_tuple[0] = model_mapping["model_id"][model_tuple[0]]["idx"]
                        new_item["eval_with"] = eval_model_tuple
                    else:
                        new_item["eval_with"] = []
                    new_item["current_language"] = [tuple_[1] for tuple_ in eval_model_tuple]
                    logger_names.append(copy.deepcopy(new_item))
                else:
                    # Create for each model a separate item
                    for eval_model in eval_model_tuple:
                        new_item["model_name"] = eval_model[0]
                        new_item["model_idx"] = model_mapping["model_id"][new_item["model_name"]]["idx"]
                        new_item["eval_with"] = []
                        new_item["model_language"] = model_mapping["model_id"][new_item["model_name"]]["languages"]
                        new_item["current_language"] = [eval_model[1]]
                        logger_names.append(copy.deepcopy(new_item))

    for item in logger_names:
        item["logger_name"] = "/".join([name_model_for_logger(item["model_language"]), stage, item["task_name"],
                                        "_".join(item["dataset"]) + "(dataset)", "_".join(item["current_language"])])

    return logger_names
