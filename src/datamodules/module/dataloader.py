from torch.utils.data.dataloader import DataLoader
from datasets.arrow_dataset import Dataset


def initialize_dataloader(data, batch_size, num_workers, pin_memory, collate_fn, specific_collate_fn):
    dataloaders = None
    if isinstance(data, dict):
        dataloader_args = {"batch_size": batch_size,
                           "num_workers": num_workers,
                           "pin_memory": pin_memory,
                           "collate_fn": specific_collate_fn
                           if specific_collate_fn is not None
                           else collate_fn,
                           # Changed: can only shuffle in map-style dataset. Will fail for iterable dataset
                           "shuffle": False}
        dataloaders = {}
        for key, value in data.items():
            dataloaders[key] = DataLoader(
                dataset=value,
                **dataloader_args
            )
    elif isinstance(data, list):
        dataloaders = []
        for index in range(len(data)):
            dataloader_args = {"dataset": get_value_for_list_or_value(data, index),
                               "batch_size": get_value_for_list_or_value(batch_size, index),
                               "num_workers": get_value_for_list_or_value(num_workers, index),
                               "pin_memory": get_value_for_list_or_value(pin_memory, index),
                               "collate_fn": get_value_for_list_or_value(specific_collate_fn, index)
                               if specific_collate_fn is not None
                               else get_value_for_list_or_value(collate_fn, index),
                               "shuffle": False}
            dataloaders.append(DataLoader(**dataloader_args))
    elif isinstance(data, Dataset):
        dataloader_args = {"batch_size": batch_size,
                           "num_workers": num_workers,
                           "pin_memory": pin_memory,
                           "collate_fn": specific_collate_fn
                           if specific_collate_fn is not None
                           else collate_fn,
                           # Changed: can only shuffle in map-style dataset. Will fail for iterable dataset
                           "shuffle": False}
        dataloaders = DataLoader(dataset=data, **dataloader_args)

    else:
        NotImplementedError(f"Please implement setup for dataloader")
    return dataloaders


def get_value_for_list_or_value(potential_list, index):
    if isinstance(potential_list, list):
        return potential_list[index]
    else:
        return potential_list
