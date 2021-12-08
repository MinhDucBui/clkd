from src.utils.utils import convert_cfg_tuple


def assert_functions(students_cfg, embedding_sharing_cfg, weight_sharing_cfg, eval_cfg):
    assert_eval_cfg(students_cfg, eval_cfg)
    assert_embedding_cfg(students_cfg, embedding_sharing_cfg)
    assert_weight_sharing_cfg(students_cfg, weight_sharing_cfg)


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
