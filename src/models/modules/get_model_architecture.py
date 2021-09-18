from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
}


def get_student_model_architecture(hparams):
    print("Initialize Student")
    # instantiate random model
    config_class, model_class, _ = MODEL_CLASSES[hparams.student_model_type]
    config = config_class.from_pretrained(hparams.student_model_name_or_path)
    config.output_hidden_states = True
    model = model_class(config)

    return model

def get_teacher_model_architecture(hparams):
    # instantiate pretrained model
    print("Initialize Teacher")
    _, model_class, _ = MODEL_CLASSES[hparams.teacher_model_type]
    model = model_class.from_pretrained(hparams.teacher_model_name_or_path, output_hidden_states=True)

    return model

def get_tokenizer(hparams):
    _, _, teacher_tokenizer_class = MODEL_CLASSES[hparams.teacher_model_type]
    tokenizer = teacher_tokenizer_class.from_pretrained(hparams.teacher_model_name_or_path)

    return tokenizer

