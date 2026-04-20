from fever_data import prepare_transformer_examples

TRAIN_PATH = "data/train.jsonl"
DEV_PATH = "data/shared_task_dev.jsonl"
WIKI_PAGES_PATH = "data/fever_wiki_pages"
PROCESSED_TRAIN_PATH = "data/processed/train_transformer.jsonl"
PROCESSED_DEV_PATH = "data/processed/dev_transformer.jsonl"

def main():
    """
    To preprocess transformer data given no arguments.
    """
    print("Preparing processed train transformer examples...")
    prepare_transformer_examples(
        TRAIN_PATH,
        WIKI_PAGES_PATH,
        PROCESSED_TRAIN_PATH,
    )
    print(f"Saved processed train examples to {PROCESSED_TRAIN_PATH}")

    print("Preparing processed dev transformer examples...")
    prepare_transformer_examples(
        DEV_PATH,
        WIKI_PAGES_PATH,
        PROCESSED_DEV_PATH,
    )
    print(f"Saved processed dev examples to {PROCESSED_DEV_PATH}")

if __name__ == "__main__":
    main()
