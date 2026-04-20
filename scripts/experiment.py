import json
import os
import numpy as np
from types import SimpleNamespace
from fever_data import load_transformer_examples, read_fever_examples
from models import train_deep_averaging_network, train_distilbert_classifier, train_logistic_regression
from utils import compute_metrics, ensure_dir, format_experiment_result, set_random_seeds

DEFAULT_CONFIG = {
    "models": ["LR", "DAN", "DISTILBERT"],
    "train_path": "data/train.jsonl",
    "dev_path": "data/shared_task_dev.jsonl",
    "processed_train_path": "data/processed/train_transformer.jsonl",
    "processed_dev_path": "data/processed/dev_transformer.jsonl",
    "word_vecs_path": "data/glove.6B.100d.txt",
    "output_path": "output/experiment_results.txt",
    "distilbert_output_dir": "artifacts/distilbert/full_run_001",
    "resume_from_checkpoint": None,
    "lr": 0.001,
    "transformer_lr": 5e-5,
    "num_epochs": 10,
    "hidden_size": 100,
    "embedding_dim": 100,
    "batch_size": 64,
    "transformer_batch_size": 16,
    "transformer_model_name": "distilbert-base-uncased",
    "max_length": 256,
    "seed": 0,
}

def evaluate_model(model, exs):
    """
    To evaluate a model given model and exs.
    """
    golds = [ex.label for ex in exs]
    preds = model.predict_all(exs)
    return compute_metrics(golds, preds)

def load_final_distilbert_metrics(config):
    """
    To load final DistilBERT dev metrics given config.
    """
    metrics_path = f"{config.distilbert_output_dir}/metrics.jsonl"
    last_record = None

    with open(metrics_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            last_record = json.loads(line)

    if last_record is None:
        raise ValueError(f"DistilBERT metrics file is empty: {metrics_path}")

    return {
        "accuracy": last_record["dev_accuracy"],
        "macro_f1": last_record["dev_macro_f1"],
        "confusion_matrix": np.array(last_record["dev_confusion_matrix"]),
    }

def build_result_config(config, model_name):
    """
    To build an experiment config given config and model_name.
    """
    if model_name == "LR":
        return {"seed": config.seed}
    if model_name == "DAN":
        return {
            "lr": config.lr,
            "num_epochs": config.num_epochs,
            "hidden_size": config.hidden_size,
            "embedding_dim": config.embedding_dim,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "word_vecs_path": config.word_vecs_path,
        }

    return {
        "transformer_lr": config.transformer_lr,
        "num_epochs": config.num_epochs,
        "transformer_batch_size": config.transformer_batch_size,
        "transformer_model_name": config.transformer_model_name,
        "max_length": config.max_length,
        "seed": config.seed,
        "processed_train_path": config.processed_train_path,
        "processed_dev_path": config.processed_dev_path,
        "distilbert_output_dir": config.distilbert_output_dir,
        "resume_from_checkpoint": config.resume_from_checkpoint,
    }

def write_results(results_text, output_path):
    """
    To write experiment results given results_text and output_path.
    """
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(results_text)

def run_baseline_experiments(config, train_exs, dev_exs, model_names):
    """
    To run baseline experiments given config, train_exs, dev_exs, and model_names.
    """
    result_blocks = []
    transformer_train_exs = None
    transformer_dev_exs = None

    for model_name in model_names:
        if model_name not in {"LR", "DAN", "DISTILBERT"}:
            raise ValueError("model_names must contain only LR, DAN, and/or DISTILBERT")

        if model_name == "LR":
            model = train_logistic_regression(train_exs)
            model_dev_exs = dev_exs
            input_setting = "claim-only"
            notes = "Baseline uses claim text only."
        elif model_name == "DAN":
            model = train_deep_averaging_network(config, train_exs, dev_exs)
            model_dev_exs = dev_exs
            input_setting = "claim-only"
            notes = "Baseline uses claim text only. Pretrained word vectors are used only if word_vecs_path is provided."
        else:
            # Load the processed transformer data once even if multiple DistilBERT runs are added later.
            if transformer_train_exs is None:
                transformer_train_exs = load_transformer_examples(config.processed_train_path)
                transformer_dev_exs = load_transformer_examples(config.processed_dev_path)
            model = train_distilbert_classifier(config, transformer_train_exs, transformer_dev_exs)
            input_setting = "claim-plus-evidence"
            notes = "Final model uses claim plus resolved evidence text."
            metrics = load_final_distilbert_metrics(config)
        if model_name != "DISTILBERT":
            metrics = evaluate_model(model, model_dev_exs)
        result_config = build_result_config(config, model_name)
        result_blocks.append(format_experiment_result(
            model_name=model_name,
            input_setting=input_setting,
            metrics=metrics,
            config=result_config,
            notes=notes))

    return "\n\n".join(result_blocks)

def main():
    """
    To run baseline comparisons given no arguments.
    """
    config = SimpleNamespace(**DEFAULT_CONFIG)
    set_random_seeds(config.seed)

    train_exs = read_fever_examples(config.train_path)
    dev_exs = read_fever_examples(config.dev_path)
    results_text = run_baseline_experiments(config, train_exs, dev_exs, config.models)
    write_results(results_text, config.output_path)
    print(f"Saved experiment results to {config.output_path}")


if __name__ == "__main__":
    main()
