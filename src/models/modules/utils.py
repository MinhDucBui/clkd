def change_embedding_layer(model, model_idx, embeddings, language):
    if model.__class__.__name__ == 'TinyModel':
        model.base.base_model.embeddings = embeddings[model_idx][language]
    else:
        model.base_model.embeddings = embeddings[model_idx][language]
