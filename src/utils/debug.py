import torch
import copy


def debug_embedding_updating(model, model_idx, batch_idx, test, test1, language):
    if model_idx == 0 and language == "ss":

        if test is not None and torch.equal(
                model[model_idx].base_model.embeddings.word_embeddings.weight, test):
            print("SAME WEIGHTS")
        else:
            print("NOT SAME WEIGHTS")
        if test1 is not None and torch.equal(
                model[model_idx].base_model.encoder.layer[0].output.dense.weight, test1):
            print("SAME WEIGHTS LAYER")
        else:
            print("NOT SAME WEIGHTS LAYER")
        if batch_idx == 0:
            test = copy.deepcopy(model[model_idx].base_model.embeddings.word_embeddings.weight)
            test1 = copy.deepcopy(model[model_idx].base_model.encoder.layer[0].output.dense.weight)


def debug_id_embeddings(embeddings):
    for embedding in embeddings:
        for language, emb in embedding.items():
            print(language, id(emb))
