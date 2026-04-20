import json
import os
from fever_data import load_transformer_examples
from models import (
    _build_transformer_dataloader,
    _evaluate_distilbert,
    _get_torch_device,
    load_distilbert_checkpoint,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import ensure_dir, format_experiment_result

DEFAULT_CONFIG_PATH = "artifacts/distilbert/full_run_001/config.json"
DEFAULT_CHECKPOINT_PATH = "artifacts/distilbert/full_run_001/best_model.pt"
DEFAULT_DEV_PATH = "data/processed/dev_transformer.jsonl"
DEFAULT_OUTPUT_PATH = "output/best_distilbert_results.txt"

def load_run_config(path):
    """
    To load a DistilBERT run config given path.
    """
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)

def build_result_config(config):
    """
    To build a DistilBERT result config given config.
    """
    return {
        "transformer_lr": config["transformer_lr"],
        "num_epochs": config["num_epochs"],
        "transformer_batch_size": config["transformer_batch_size"],
        "transformer_model_name": config["transformer_model_name"],
        "max_length": config["max_length"],
        "seed": config["seed"],
        "processed_train_path": config.get("processed_train_path"),
        "processed_dev_path": config.get("processed_dev_path"),
        "distilbert_output_dir": config["output_dir"],
        "resume_from_checkpoint": config.get("resume_from_checkpoint"),
    }

def load_best_distilbert_model(config, checkpoint_path):
    """
    To load the best DistilBERT model given config and checkpoint_path.
    """
    tokenizer = AutoTokenizer.from_pretrained(config["transformer_model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["transformer_model_name"],
        num_labels=3,
    )
    device = _get_torch_device()
    model = model.to(device)
    load_distilbert_checkpoint(checkpoint_path, model)
    return model, tokenizer, device

def evaluate_best_distilbert(model, tokenizer, device, dev_exs, config):
    """
    To evaluate the best DistilBERT model given model, tokenizer, device, dev_exs, and config.
    """
    dev_dataloader = _build_transformer_dataloader(
        dev_exs,
        tokenizer,
        config["max_length"],
        config["transformer_batch_size"],
        False,
    )
    return _evaluate_distilbert(model, dev_dataloader, device)

def write_best_distilbert_results(output_path, metrics, config):
    """
    To write best DistilBERT results given output_path, metrics, and config.
    """
    result_text = format_experiment_result(
        model_name="DISTILBERT",
        input_setting="claim-plus-evidence",
        metrics=metrics,
        config=build_result_config(config),
        notes="Metrics computed by loading best_model.pt and re-running dev inference.",
    )

    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(result_text)

def main():
    """
    To evaluate the best saved DistilBERT checkpoint given no arguments.
    """
    config = load_run_config(DEFAULT_CONFIG_PATH)
    dev_exs = load_transformer_examples(DEFAULT_DEV_PATH)
    model, tokenizer, device = load_best_distilbert_model(config, DEFAULT_CHECKPOINT_PATH)
    metrics = evaluate_best_distilbert(model, tokenizer, device, dev_exs, config)
    write_best_distilbert_results(DEFAULT_OUTPUT_PATH, metrics, config)
    print(f"Saved best DistilBERT results to {DEFAULT_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
