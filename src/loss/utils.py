

def create_hidden_attention_mapping(teacher_outputs, student_outputs, layer_mapping):

    mapping_factor = int((len(teacher_outputs['hidden_states']) - 1) / (len(student_outputs['hidden_states']) - 1))

    # Assure that layers can be mapped
    assert (len(
        teacher_outputs[
            'hidden_states']) - 1) % mapping_factor == 0, "Not able to map teacher layers to student layers. " \
                                                          "Change the number of student's layers."

    if not layer_mapping:
        # Create mapping dict
        mapping_hid = {0: 0}
        for i in range(1, len(student_outputs['hidden_states'])):
            mapping_hid[i] = i * mapping_factor

        mapping_att = mapping_hid[1:]
        mapping_att = [(x - 1, y - 1) for (x, y) in mapping_att]

    else:
        mapping_hid = layer_mapping
        mapping_att = {key - 1: value - 1 for (key, value) in mapping_hid.items()}

    return mapping_hid, mapping_att