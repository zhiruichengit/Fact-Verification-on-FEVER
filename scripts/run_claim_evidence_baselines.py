from types import SimpleNamespace
from nltk.tokenize import TreebankWordTokenizer
from fever_data import load_transformer_examples
from models import train_deep_averaging_network, train_logistic_regression
from scripts.experiment import DEFAULT_CONFIG, evaluate_model, write_results
from utils import format_experiment_result, set_random_seeds

TOKENIZER = TreebankWordTokenizer()

DEFAULT_CLAIM_EVIDENCE_CONFIG = {
    **DEFAULT_CONFIG,
    "models": ["LR", "DAN"],
    "output_path": "output/claim_evidence_baselines_results.txt",
}

def combine_claim_and_evidence(claim, evidence_text):
    """
    To combine claim and evidence text into one baseline input string.
    """
    claim_text = claim.strip() if claim is not None else ""
    evidence = evidence_text.strip() if evidence_text is not None else ""

    if evidence:
        return f"{claim_text} [SEP] {evidence}"
    return claim_text


def tokenize_text(text):
    """
    To tokenize text for DAN input.
    """
    tokens = []
    for word in TOKENIZER.tokenize(text.lower()):
        if word:
            tokens.append(word)
    return tokens

def convert_transformer_examples_to_claim_evidence_examples(transformer_examples):
    """
    To convert processed transformer examples into baseline-compatible claim+evidence examples.
    """
    baseline_examples = []

    for ex in transformer_examples:
        combined_text = combine_claim_and_evidence(ex.claim, ex.evidence_text)
        baseline_examples.append(
            SimpleNamespace(
                claim=combined_text,
                label=ex.label,
                words=tokenize_text(combined_text),
            )
        )

    return baseline_examples

def build_claim_evidence_result_config(config, model_name):
    """
    To build a result config block for claim+evidence baseline experiments.
    """
    if model_name == "LR":
        return {
            "seed": config.seed,
            "processed_train_path": config.processed_train_path,
            "processed_dev_path": config.processed_dev_path,
            "input_representation": "claim [SEP] evidence_text",
        }

    return {
        "lr": config.lr,
        "num_epochs": config.num_epochs,
        "hidden_size": config.hidden_size,
        "embedding_dim": config.embedding_dim,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "word_vecs_path": config.word_vecs_path,
        "processed_train_path": config.processed_train_path,
        "processed_dev_path": config.processed_dev_path,
        "input_representation": "claim [SEP] evidence_text",
    }

def run_claim_evidence_baseline_experiments(config):
    """
    To run LR and DAN experiments using claim plus evidence text.
    """
    transformer_train_exs = load_transformer_examples(config.processed_train_path)
    transformer_dev_exs = load_transformer_examples(config.processed_dev_path)

    train_exs = convert_transformer_examples_to_claim_evidence_examples(transformer_train_exs)
    dev_exs = convert_transformer_examples_to_claim_evidence_examples(transformer_dev_exs)

    result_blocks = []

    for model_name in config.models:
        if model_name not in {"LR", "DAN"}:
            raise ValueError("Claim+evidence baseline script supports only LR and DAN")

        if model_name == "LR":
            model = train_logistic_regression(train_exs)
            notes = "Baseline uses concatenated claim and resolved evidence text."
        else:
            model = train_deep_averaging_network(config, train_exs, dev_exs)
            notes = (
                "Baseline uses concatenated claim and resolved evidence text. "
                "Pretrained word vectors are used only if word_vecs_path is provided."
            )

        metrics = evaluate_model(model, dev_exs)
        result_config = build_claim_evidence_result_config(config, model_name)

        result_blocks.append(
            format_experiment_result(
                model_name=model_name,
                input_setting="claim-plus-evidence",
                metrics=metrics,
                config=result_config,
                notes=notes))

    return "\n\n".join(result_blocks)

def main():
    """
    To run claim+evidence LR and DAN experiments given no arguments.
    """
    config = SimpleNamespace(**DEFAULT_CLAIM_EVIDENCE_CONFIG)
    set_random_seeds(config.seed)

    results_text = run_claim_evidence_baseline_experiments(config)
    write_results(results_text, config.output_path)
    print(f"Saved claim+evidence baseline results to {config.output_path}")

if __name__ == "__main__":
    main()