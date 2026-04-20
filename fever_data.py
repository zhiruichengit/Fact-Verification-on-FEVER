import json
import os
import numpy as np
import torch
from nltk.tokenize import TreebankWordTokenizer
from utils import Indexer, ensure_dir
from datasets import load_from_disk

LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}

TOKENIZER = TreebankWordTokenizer()

def _tokenize_claim(claim):
    """
    To tokenize a claim string given claim.
    """
    tokens = []
    for word in TOKENIZER.tokenize(claim.lower()):
        if word:
            tokens.append(word)
    return tokens

class FeverExample:
    """
    Represents a single FEVER claim example.
    """

    def __init__(self, claim, label):
        """
        To initialize a FEVER example given claim and label.
        """
        self.claim = claim
        self.label = label
        self.words = _tokenize_claim(claim)

    def __repr__(self):
        """
        To return a string representation given no arguments.
        """
        return repr(self.claim) + "; label=" + repr(self.label)

class TransformerExample:
    """
    Represents a transformer example for FEVER claims and evidence.
    """

    def __init__(self, claim, evidence_text, label):
        """
        To initialize a transformer example given claim, evidence_text, and label.
        """
        self.claim = claim
        self.evidence_text = evidence_text
        self.label = label

    def __repr__(self):
        """
        To return a string representation given no arguments.
        """
        return repr(self.claim) + "; evidence=" + repr(self.evidence_text) + "; label=" + repr(self.label)

class WordEmbeddings:
    """
    Represents a set of word embeddings and a word indexer.
    """

    def __init__(self, word_indexer, vectors):
        """
        To initialize word embeddings given word_indexer and vectors.
        """
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=True, padding_idx=0):
        """
        To return an embedding layer given frozen and padding_idx.
        """
        return torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(self.vectors),
            freeze=frozen,
            padding_idx=padding_idx,
        )

    def get_embedding_length(self):
        """
        To return the embedding length given no arguments.
        """
        return len(self.vectors[0])

def read_fever_examples(path):
    """
    To read FEVER examples given path.
    """
    examples = []

    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(FeverExample(record["claim"], LABEL_MAP[record["label"]]))

    return examples

def read_fever_records(path):
    """
    To read raw FEVER records given path.
    """
    records = []

    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            records.append(json.loads(line))

    return records

def read_blind_fever_examples(path):
    """
    To read blind FEVER examples given path.
    """
    examples = []

    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(FeverExample(record["claim"], None))

    return examples

def read_wiki_pages(path):
    """
    To read wiki pages given path.
    """
    return load_from_disk(path)

def normalize_wiki_title(title):
    """
    To normalize a wiki title given title.
    """
    if title is None:
        return ""
    return " ".join(str(title).strip().split()).replace(" ", "_")

def build_wiki_title_index(wiki_pages):
    """
    To build a wiki title index given wiki_pages.
    """
    wiki_split = _get_wiki_split(wiki_pages)
    wiki_index = {}

    for row_idx in range(len(wiki_split)):
        row = wiki_split[row_idx]
        normalized_title = normalize_wiki_title(row["id"])
        # Keep the first page we see for each normalized title.
        if not normalized_title or normalized_title in wiki_index:
            continue
        wiki_index[normalized_title] = {
            "id": row["id"],
            "text": row["text"],
            "lines": _parse_wiki_lines(row["lines"]),
        }

    return wiki_index

def resolve_evidence_text(record, wiki_index):
    """
    To resolve evidence text given record and wiki_index.
    """
    # FEVER can provide multiple gold evidence sets; use the first one we can resolve.
    for evidence_set in record.get("evidence", []):
        resolved_sentences = []

        for evidence_ref in evidence_set:
            if len(evidence_ref) < 4:
                continue
            wiki_title = normalize_wiki_title(evidence_ref[2])
            line_number = evidence_ref[3]
            if not wiki_title or line_number is None:
                continue
            wiki_record = wiki_index.get(wiki_title)
            if wiki_record is None:
                continue
            sentence_text = wiki_record["lines"].get(int(line_number))
            if sentence_text:
                resolved_sentences.append(sentence_text)

        # Concatenate one evidence set in order to keep the input deterministic.
        if resolved_sentences:
            return " ".join(resolved_sentences)

    return ""

def build_transformer_examples(records, wiki_index):
    """
    To build transformer examples given records and wiki_index.
    """
    examples = []

    for record in records:
        label = LABEL_MAP[record["label"]] if record.get("label") is not None else None
        evidence_text = resolve_evidence_text(record, wiki_index)
        examples.append(TransformerExample(record["claim"], evidence_text, label))

    return examples

def write_transformer_examples(examples, path):
    """
    To write transformer examples given examples and path.
    """
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as outfile:
        for example in examples:
            record = {
                "claim": example.claim,
                "evidence_text": example.evidence_text,
                "label": example.label,
            }
            outfile.write(json.dumps(record) + "\n")

def load_transformer_examples(path):
    """
    To load transformer examples given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed transformer examples not found at {path}. Run python -m scripts.prepare_transformer_data first."
        )

    examples = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(TransformerExample(record["claim"], record["evidence_text"], record["label"]))

    return examples

def prepare_transformer_examples(fever_path, wiki_pages_path, output_path):
    """
    To prepare transformer examples given fever_path, wiki_pages_path, and output_path.
    """
    records = read_fever_records(fever_path)
    wiki_pages = read_wiki_pages(wiki_pages_path)
    evidence_titles = _collect_evidence_titles(records)
    # Filter the wiki dump before indexing so preprocessing stays manageable.
    filtered_wiki_pages = _filter_wiki_pages_by_titles(wiki_pages, evidence_titles)
    wiki_index = build_wiki_title_index(filtered_wiki_pages)
    examples = build_transformer_examples(records, wiki_index)
    write_transformer_examples(examples, output_path)

def read_transformer_examples(path, wiki_pages_path):
    """
    To read transformer examples given path and wiki_pages_path.
    """
    records = read_fever_records(path)
    wiki_pages = read_wiki_pages(wiki_pages_path)
    evidence_titles = _collect_evidence_titles(records)
    filtered_wiki_pages = _filter_wiki_pages_by_titles(wiki_pages, evidence_titles)
    wiki_index = build_wiki_title_index(filtered_wiki_pages)
    return build_transformer_examples(records, wiki_index)

def read_word_embeddings(embeddings_file):
    """
    To read word embeddings given embeddings_file.
    """
    infile = open(embeddings_file, "r", encoding="utf-8")
    word_indexer = Indexer()
    vectors = []

    for line in infile:
        if line.strip() != "":
            space_idx = line.find(" ")
            word = line[:space_idx]
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    infile.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    return WordEmbeddings(word_indexer, np.array(vectors))

def _get_wiki_split(wiki_pages):
    """
    To get the wiki split given wiki_pages.
    """
    if "wikipedia_pages" in wiki_pages:
        return wiki_pages["wikipedia_pages"]
    if hasattr(wiki_pages, "keys"):
        first_key = list(wiki_pages.keys())[0]
        return wiki_pages[first_key]
    return wiki_pages

def _collect_evidence_titles(records):
    """
    To collect evidence titles given records.
    """
    titles = set()

    for record in records:
        for evidence_set in record.get("evidence", []):
            for evidence_ref in evidence_set:
                # Ignore malformed references and keep only titles that can be looked up.
                if len(evidence_ref) < 4 or evidence_ref[2] is None:
                    continue
                titles.add(normalize_wiki_title(evidence_ref[2]))

    return titles

def _filter_wiki_pages_by_titles(wiki_pages, titles):
    """
    To filter wiki pages given wiki_pages and titles.
    """
    wiki_split = _get_wiki_split(wiki_pages)
    if not titles:
        return wiki_split.select([])

    return wiki_split.filter(
        lambda batch: [normalize_wiki_title(title) in titles for title in batch["id"]],
        batched=True,
        batch_size=10000,
    )

def _parse_wiki_lines(lines_text):
    """
    To parse wiki lines given lines_text.
    """
    parsed_lines = {}
    if lines_text is None:
        return parsed_lines

    for raw_line in lines_text.split("\n"):
        if not raw_line:
            continue
        line_parts = raw_line.split("\t")
        if not line_parts or not line_parts[0].isdigit():
            continue
        sentence_text = line_parts[1].strip() if len(line_parts) > 1 else ""
        if sentence_text:
            parsed_lines[int(line_parts[0])] = sentence_text

    return parsed_lines
