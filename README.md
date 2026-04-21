# Fact Verification on FEVER

This repository implements a fact verification pipeline for the FEVER dataset. It treats FEVER as a three-way classification task with labels `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO`, and compares three model families:

- Logistic Regression (LR) as a TF-IDF lexical baseline
- a Deep Averaging Network (DAN) over pretrained GloVe embeddings
- DistilBERT as a pretrained paired-text classifier

The project studies both claim-only classification and claim-and-evidence classification. Experiment summaries are written to `output/`.

## Data

The code expects FEVER resources under `data/`:

- `data/train.jsonl`
- `data/shared_task_dev.jsonl`
- `data/fever_wiki_pages`
- `data/glove.6B.100d.txt`

Download the data folder here `https://drive.google.com/drive/folders/1mz9uReSwXAZIsNUGF2lxW-vcFBc6NMSV?usp=sharing`

The dataset is moderately imbalanced in training:

- `SUPPORTS`: 80,035 examples
- `NOT ENOUGH INFO`: 35,639 examples
- `REFUTES`: 29,775 examples

The development split is balanced at 6,666 examples per class.

Raw FEVER examples follow the FEVER JSONL format. For example:

```json
{
  "id": 137334,
  "verifiable": "VERIFIABLE",
  "label": "SUPPORTS",
  "claim": "Fox 2000 Pictures released the film Soul Food.",
  "evidence": [
    [[289914, 283015, "Soul_Food_-LRB-film-RRB-", 0]],
    [[291259, 284217, "Soul_Food_-LRB-film-RRB-", 0]],
    [[293412, 285960, "Soul_Food_-LRB-film-RRB-", 0]],
    [[337212, 322620, "Soul_Food_-LRB-film-RRB-", 0]],
    [[337214, 322622, "Soul_Food_-LRB-film-RRB-", 0]]
  ]
}
```

For claim-only models, the code uses the `claim` and `label` fields. For evidence-aware preprocessing, it also uses FEVER `evidence` annotations together with the Wikipedia pages dataset stored at `data/fever_wiki_pages`.

Processed claim-evidence examples are written as JSONL with integer labels. For example:

```json
{
  "claim": "Fox 2000 Pictures released the film Soul Food.",
  "evidence_text": "Soul Food is a 1997 American comedy-drama film produced by Kenneth `` Babyface '' Edmonds , Tracey Edmonds and Robert Teitel and released by Fox 2000 Pictures .",
  "label": 0
}
```

Processed label mapping from `fever_data.py`:

- `0 = SUPPORTS`
- `1 = REFUTES`
- `2 = NOT ENOUGH INFO`

For `NOT ENOUGH INFO` examples, `evidence_text` may be empty.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## How the Pipeline Works

### Claim-only baselines

`claim_verifier.py` trains two claim-only baselines:

- `LR`: TF-IDF features with logistic regression
- `DAN`: a deep averaging network that averages pretrained GloVe token embeddings and classifies the averaged representation

These models predict FEVER labels from the claim text alone.

### Claim-and-evidence preprocessing

`scripts/prepare_transformer_data.py` builds processed claim-evidence examples for evidence-aware experiments:

1. Read FEVER records from `train.jsonl` or `shared_task_dev.jsonl`.
2. Collect Wikipedia page titles referenced in FEVER evidence annotations.
3. Filter the saved Wikipedia dataset down to those titles.
4. Resolve sentence-level evidence text from page titles and line numbers.
5. Write processed JSONL records containing `claim`, `evidence_text`, and an integer `label`.

The processed files are:

- `data/processed/train_transformer.jsonl`
- `data/processed/dev_transformer.jsonl`

### Evidence-aware models

The project evaluates two evidence-aware settings:

- evidence-aware LR and DAN baselines trained on a concatenated input of the form `claim [SEP] evidence_text`
- DistilBERT fine-tuned as a three-way classifier on paired `(claim, evidence_text)` inputs using Hugging Face tokenization and pretrained sequence classification weights

The processed claim-evidence examples are reused for both DistilBERT and the evidence-aware LR/DAN baselines so preprocessing stays consistent across those experiments.

## Running the Project

Run the main training and evaluation entry point from the repository root:

```bash
python claim_verifier.py --model LR
```

Available model choices:

- `LR`
- `DAN`
- `DISTILBERT`

Common arguments:

- `--train_path`: defaults to `data/train.jsonl`
- `--dev_path`: defaults to `data/shared_task_dev.jsonl`
- `--word_vecs_path`: defaults to `data/glove.6B.100d.txt`
- `--num_epochs`: defaults to `10`
- `--batch_size`: defaults to `64`
- `--seed`: defaults to `0`

DistilBERT-specific arguments:

- `--processed_train_path`: defaults to `data/processed/train_transformer.jsonl`
- `--processed_dev_path`: defaults to `data/processed/dev_transformer.jsonl`
- `--transformer_model_name`: defaults to `distilbert-base-uncased`
- `--transformer_lr`: defaults to `5e-5`
- `--transformer_batch_size`: defaults to `16`
- `--max_length`: defaults to `256`
- `--output_dir`: defaults to `artifacts/distilbert/run_001`
- `--resume_from_checkpoint`: optional checkpoint path

Example commands:

```bash
python claim_verifier.py --model LR
python claim_verifier.py --model DAN
python claim_verifier.py --model DISTILBERT
```

## Experiments and Evaluation

The project reports both accuracy and macro-F1. Accuracy gives overall performance, while macro-F1 is included because the training labels are not perfectly balanced and we want a metric that better reflects performance across all three classes.

### Prepare transformer data

```bash
python -m scripts.prepare_transformer_data
```

This writes:

- `data/processed/train_transformer.jsonl`
- `data/processed/dev_transformer.jsonl`

### Run the main experiment comparison

```bash
python -m scripts.experiment
```

This writes `output/experiment_results.txt`, which summarizes the claim-only LR and DAN baselines together with the DistilBERT claim-and-evidence run.

### Run claim-and-evidence baselines

```bash
python -m scripts.run_claim_evidence_baselines
```

This writes `output/claim_evidence_baselines_results.txt`, which evaluates LR and DAN on concatenated `claim [SEP] evidence_text` inputs.

### Evaluate the best saved DistilBERT checkpoint

```bash
python -m scripts.evaluate_best_distilbert
```

This script expects generated artifacts from a prior DistilBERT training run, typically under a path such as `artifacts/distilbert/full_run_001/`, and writes `output/best_distilbert_results.txt`.

## Results

Table 1 summarizes the development-set results reported in the project report. DistilBERT is reported using the best saved checkpoint.

| Model | Feature | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| LR | Claim-only | 0.4499 | 0.4170 |
| DAN | Claim-only | 0.4313 | 0.3730 |
| LR | Claim and Evidence | 0.7164 | 0.6774 |
| DAN | Claim and Evidence | 0.6592 | 0.5611 |
| DistilBERT (2nd epoch) | Claim and Evidence | 0.9336 | 0.9333 |

## Key Takeaways

- Adding evidence substantially improves the shallow baselines over the claim-only setting.
- Claim-and-evidence LR performs better than claim-and-evidence DAN in this project.
- Both shallow evidence-aware baselines still struggle on `REFUTES`; the smaller `REFUTES` training class likely contributes, but contradiction modeling remains the main weakness.
- DistilBERT is the strongest model because paired-text transformer modeling captures richer claim-evidence interactions than the shallow baselines.

## Repository Layout

```text
claim_verifier.py                       Main training and evaluation entry point
fever_data.py                           FEVER readers, wiki indexing, evidence resolution
models.py                               LR, DAN, and DistilBERT implementations
utils.py                                Metrics and helper utilities
scripts/prepare_transformer_data.py     Preprocess claim-evidence examples
scripts/experiment.py                   Main experiment runner
scripts/run_claim_evidence_baselines.py Claim-and-evidence LR/DAN experiments
scripts/evaluate_best_distilbert.py     Evaluate the best saved DistilBERT checkpoint
output/                                 Experiment summaries
eda.ipynb                               Exploratory notebook
```
