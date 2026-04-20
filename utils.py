import os
import random
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

class Indexer:
    """
    Represents a bidirectional mapping between objects and indices.
    """

    def __init__(self):
        """
        To initialize an indexer given no arguments.
        """
        self.objs_to_ints = {}
        self.ints_to_objs = {}
        self.add_and_get_index("PAD")
        self.add_and_get_index("UNK")

    def __len__(self):
        """
        To return the size of the indexer given no arguments.
        """
        return len(self.objs_to_ints)

    def add_and_get_index(self, obj):
        """
        To add an object and return its index given obj.
        """
        if obj not in self.objs_to_ints:
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[obj] = new_idx
            self.ints_to_objs[new_idx] = obj
        return self.objs_to_ints[obj]

    def index_of(self, obj):
        """
        To return the index of an object given obj.
        """
        return self.objs_to_ints.get(obj, self.objs_to_ints["UNK"])

    def get_object(self, idx):
        """
        To return the object at an index given idx.
        """
        return self.ints_to_objs.get(idx)

    def contains(self, obj):
        """
        To check whether an object is present given obj.
        """
        return obj in self.objs_to_ints

def pad_batch(index_batches, pad_value=0):
    """
    To pad a batch of index lists given index_batches and pad_value.
    """
    max_len = max(len(indices) for indices in index_batches)
    return [indices + [pad_value] * (max_len - len(indices)) for indices in index_batches]

def form_input(x):
    """
    To convert values to a tensor given x.
    """
    return torch.tensor(x, dtype=torch.long)

def set_random_seeds(seed):
    """
    To set random seeds given seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    """
    To ensure a directory exists given path.
    """
    if path:
        os.makedirs(path, exist_ok=True)

def compute_metrics(golds, preds):
    """
    To compute evaluation metrics given golds and preds.
    """
    return {
        "accuracy": accuracy_score(golds, preds),
        "macro_f1": f1_score(golds, preds, average="macro"),
        "confusion_matrix": confusion_matrix(golds, preds, labels=[0, 1, 2]),
    }

def format_confusion_matrix(confusion_mat):
    """
    To format a confusion matrix given confusion_mat.
    """
    row_label_width = 18
    col_width = 18

    header_cells = ["gold\\pred"] + LABEL_NAMES
    header = " | ".join(f"{cell:<{col_width}}" for cell in header_cells)
    separator = "-" * len(header)

    lines = [header, separator]

    for label_name, row in zip(LABEL_NAMES, confusion_mat):
        row_cells = [label_name] + [str(int(value)) for value in row]
        line = " | ".join(f"{cell:<{col_width}}" for cell in row_cells)
        lines.append(line)

    return "\n".join(lines)

def format_experiment_result(model_name, input_setting, metrics, config, notes=""):
    """
    To format an experiment result block given model_name, input_setting, metrics, config, and notes.
    """
    lines = [f"Model: {model_name}", f"Input Setting: {input_setting}", "Config:"]

    for key, value in config.items():
        lines.append(f"{key}: {value}")

    lines.extend([
        f"Label Accuracy: {metrics['accuracy']:.4f}",
        f"Macro-F1: {metrics['macro_f1']:.4f}",
        "Confusion Matrix:",
        format_confusion_matrix(metrics["confusion_matrix"]),
    ])

    if notes:
        lines.append(f"Notes: {notes}")

    return "\n".join(lines)
