import argparse
from fever_data import load_transformer_examples, read_fever_examples
from models import train_deep_averaging_network, train_distilbert_classifier, train_logistic_regression
from utils import compute_metrics, format_confusion_matrix

def _parse_args():
    """
    To parse command-line arguments given no arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LR", choices=["LR", "DAN", "DISTILBERT"])
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--dev_path", type=str, default="data/shared_task_dev.jsonl")
    parser.add_argument("--processed_train_path", type=str, default="data/processed/train_transformer.jsonl")
    parser.add_argument("--processed_dev_path", type=str, default="data/processed/dev_transformer.jsonl")
    parser.add_argument("--word_vecs_path", type=str, default="data/glove.6B.100d.txt")
    parser.add_argument("--output_dir", type=str, default="artifacts/distilbert/run_001")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--transformer_lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--transformer_batch_size", type=int, default=16)
    parser.add_argument("--transformer_model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def evaluate(model, exs):
    """
    To evaluate a model given model and exs.
    """
    golds = [ex.label for ex in exs]
    preds = model.predict_all(exs)
    metrics = compute_metrics(golds, preds)

    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Confusion Matrix:")
    print(format_confusion_matrix(metrics["confusion_matrix"]))
    return metrics

if __name__ == "__main__":
    args = _parse_args()

    if args.model == "DISTILBERT":
        train_exs = load_transformer_examples(args.processed_train_path)
        dev_exs = load_transformer_examples(args.processed_dev_path)
    else:
        train_exs = read_fever_examples(args.train_path)
        dev_exs = read_fever_examples(args.dev_path)

    print("train examples:", len(train_exs))
    print("dev examples:", len(dev_exs))
    print("model:", args.model)

    if args.model == "LR":
        model = train_logistic_regression(train_exs)
    elif args.model == "DAN":
        model = train_deep_averaging_network(args, train_exs, dev_exs)
    else:
        model = train_distilbert_classifier(args, train_exs, dev_exs)

    evaluate(model, dev_exs)
