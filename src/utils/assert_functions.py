from src.utils.utils import convert_cfg_tuple, initialize_evaluation_cfg
from src.utils.mappings import create_model_mapping, create_validation_mapping
import copy


def assert_functions(cfg):
    # Extract configs
    students_cfg = {}
    for model_name, model_cfg in cfg.students.individual.items():
        if "student_" not in model_name:
            continue
        students_cfg[model_name] = model_cfg
    embedding_sharing_cfg = cfg.students.embed_sharing
    weight_sharing_cfg = cfg.students.weight_sharing_across_students
    eval_cfg = cfg["evaluation"]
    eval_cfg = initialize_evaluation_cfg(eval_cfg)
    if "callbacks" in cfg.keys() and "model_checkpoint" in cfg["callbacks"].keys():
        model_checkpoint_cfg = cfg["callbacks"]["model_checkpoint"]
        assert_model_checkpoint_cfg(students_cfg, eval_cfg, model_checkpoint_cfg)

    # Assertion Functions
    assert_datamodule_cfg(cfg.datamodule)
    assert_eval_cfg(students_cfg, eval_cfg)
    assert_embedding_cfg(students_cfg, embedding_sharing_cfg)
    assert_weight_sharing_cfg(students_cfg, weight_sharing_cfg)


def assert_datamodule_cfg(data_cfg):
    for key, value in data_cfg.items():
        assert "_target_" in value.keys(), "No _target_ specified in datamodule {}".format(key)


def assert_model_checkpoint_cfg(students_cfg, eval_cfg, model_checkpoint_cfg):
    monitor_name = model_checkpoint_cfg["monitor"]
    model_mapping = create_model_mapping(students_cfg)
    logger_names = create_validation_mapping(eval_cfg, model_mapping, stage="val")
    correct_logging_name = False
    all_log_names = []
    for log_name in logger_names:
        log_name = log_name["logger_name"] + "/" + log_name["metric_name"]
        if log_name == monitor_name:
            correct_logging_name = True
        all_log_names.append(log_name)
    assert correct_logging_name, \
        "ModelCheckpoint(monitor='{}') not found in the returned metrics: {}.".format(monitor_name, all_log_names)


def assert_eval_cfg(students_cfg, eval_cfg):
    for eval_name, cfg in eval_cfg.items():
        for single_tuple in cfg["evaluate_with"]:
            for model in single_tuple:
                if model[0] == "teacher":
                    continue
                assert model[0] in students_cfg.keys(), \
                    "Model {} in evaluation config is not defined in students_cfg".format(model[0])
                assert model[1] in students_cfg[model[0]]["languages"], \
                    "Language {} for Model {} in evaluation config is not defined in students_cfg".format(model[1],
                                                                                                          model[0])


def assert_embedding_cfg(students_cfg, embedding_sharing_cfg):
    options = ["in_each_model", "in_overlapping_language"]
    new_cfg = []
    if isinstance(embedding_sharing_cfg, str):
        assert embedding_sharing_cfg in options, \
            "Specified option '{}' in embed_sharing is not implemented. " \
            "Please choose one of the following options: {}".format(embedding_sharing_cfg, options)
    else:
        for embedding_sharing_tuple in embedding_sharing_cfg:
            new_cfg.append(convert_cfg_tuple(embedding_sharing_tuple))
        for single_tuple in new_cfg:
            for model in single_tuple:
                assert model[0] in students_cfg.keys(), \
                    "Model {} in embed_sharing is not defined in students_cfg".format(model[0])
                assert model[1] in students_cfg[model[0]]["languages"], \
                    "Language {} for Model {} in embed_sharing is not defined in students_cfg".format(model[1],
                                                                                                      model[0])


def assert_weight_sharing_cfg(students_cfg, weight_sharing_cfg):
    options = [False]
    new_cfg = []
    if isinstance(weight_sharing_cfg, (str, bool)):
        assert weight_sharing_cfg in options, \
            "Specified option '{}' in embed_sharing is not implemented. " \
            "Please choose one of the following options: {}".format(weight_sharing_cfg, options)

    else:
        for weight_sharing_tuple in weight_sharing_cfg:
            new_cfg.append(convert_cfg_tuple(weight_sharing_tuple))
        for single_tuple in new_cfg:
            for model in single_tuple:
                assert model[0] in students_cfg.keys(), \
                    "Model {} in weight_sharing is not defined in students_cfg".format(model[0])


def assert_loss(base_loss, loss_weighting):
    for loss_name, loss in base_loss.items():
        assert loss_name in loss_weighting.keys(), \
            "Loss {} defined but no loss_weighting found in loss cfg".format(loss_name)

    for loss_name, _ in loss_weighting.items():
        assert loss_name in loss_weighting.keys(), \
            "loss_weighting {} defined but not defined in base_loss in loss cfg".format(loss_name)
