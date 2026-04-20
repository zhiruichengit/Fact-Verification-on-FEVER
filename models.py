import json
import os
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from fever_data import read_word_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import Indexer, compute_metrics, ensure_dir, form_input, pad_batch, set_random_seeds
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ClaimClassifier(object):
    """
    Represents a base classifier for FEVER claims.
    """

    def predict(self, ex):
        """
        To predict a label given ex.
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, exs):
        """
        To predict labels for examples given exs.
        """
        predictions = []
        for ex in exs:
            predictions.append(self.predict(ex))
        return predictions

class LogisticRegressionClassifier(ClaimClassifier):
    """
    Represents a logistic regression classifier for FEVER claims.
    """

    def __init__(self, vectorizer, model):
        """
        To initialize a logistic regression classifier given vectorizer and model.
        """
        self.vectorizer = vectorizer
        self.model = model

    def predict(self, ex):
        """
        To predict a label given ex.
        """
        x = self.vectorizer.transform([ex.claim])
        return int(self.model.predict(x)[0])

    def predict_all(self, exs):
        """
        To predict labels for examples given exs.
        """
        claims = []
        for ex in exs:
            claims.append(ex.claim)
        x = self.vectorizer.transform(claims)

        predictions = []
        for pred in self.model.predict(x):
            predictions.append(int(pred))
        return predictions

class DeepAveragingNetwork(nn.Module):
    """
    Represents a deep averaging network for FEVER claims.
    """

    def __init__(self, embedding_layer, hidden_size, num_classes):
        """
        To initialize a deep averaging network given embedding_layer, hidden_size, and num_classes.
        """
        super().__init__()
        self.embedding_layer = embedding_layer
        self.embedding_dim = embedding_layer.embedding_dim
        self.hidden = nn.Linear(self.embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)

    def _avg_embedded_words(self, embedded_words, word_indices):
        """
        To average embedded words given embedded_words and word_indices.
        """
        avg_embeddings = []
        for batch_idx in range(embedded_words.size(0)):
            # Ignore PAD tokens when averaging the claim representation.
            not_pad = word_indices[batch_idx] != 0
            real_words = embedded_words[batch_idx][not_pad]
            if real_words.size(0) == 0:
                avg_embeddings.append(torch.zeros(self.embedding_dim, device=embedded_words.device))
            else:
                avg_embeddings.append(real_words.mean(dim=0))
        return torch.stack(avg_embeddings)

    def forward(self, word_indices):
        """
        To compute logits given word_indices.
        """
        embeddings = self.embedding_layer(word_indices)
        avg_embeddings = self._avg_embedded_words(embeddings, word_indices)
        hidden = self.relu(self.hidden(avg_embeddings))
        return self.output(hidden)

class NeuralClaimClassifier(ClaimClassifier):
    """
    Represents a neural classifier wrapper for FEVER claims.
    """

    def __init__(self, model, indexer, device, batch_size):
        """
        To initialize a neural claim classifier given model, indexer, device, and batch_size.
        """
        self.model = model
        self.indexer = indexer
        self.device = device
        self.batch_size = batch_size

    def predict(self, ex):
        """
        To predict a label given ex.
        """
        return self.predict_all([ex])[0]

    def predict_all(self, exs):
        """
        To predict labels for examples given exs.
        """
        if not exs:
            return []

        self.model.eval()
        predictions = []

        with torch.no_grad():
            # Batch inference keeps prediction memory usage stable on longer splits.
            for start_idx in range(0, len(exs), self.batch_size):
                batch_exs = exs[start_idx:start_idx + self.batch_size]
                batch_word_indices = []
                for ex in batch_exs:
                    batch_word_indices.append(_words_to_indices(ex.words, self.indexer))
                batch_tensor = form_input(pad_batch(batch_word_indices)).to(self.device)
                logits = self.model(batch_tensor)
                predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        return predictions

class DistilBertClaimClassifier(ClaimClassifier):
    """
    Represents a DistilBERT classifier for FEVER claims with evidence.
    """

    def __init__(self, model, tokenizer, device, max_length):
        """
        To initialize a DistilBERT classifier given model, tokenizer, device, and max_length.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def predict(self, ex):
        """
        To predict a label given ex.
        """
        return self.predict_all([ex])[0]

    def predict_all(self, exs):
        """
        To predict labels for examples given exs.
        """
        if not exs:
            return []

        dataloader = _build_transformer_dataloader(exs, self.tokenizer, self.max_length, 16, False)
        predictions = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                batch = _move_transformer_batch_to_device(batch, self.device)
                # The prediction path does not need labels, even if the dataloader includes them.
                batch.pop("labels", None)
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        return predictions

def train_logistic_regression(train_exs):
    """
    To train a logistic regression classifier given train_exs.
    """
    train_claims = []
    train_labels = []
    for ex in train_exs:
        train_claims.append(ex.claim)
        train_labels.append(ex.label)

    vectorizer = _build_tfidf_vectorizer()
    train_x = vectorizer.fit_transform(train_claims)
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(train_x, train_labels)
    return LogisticRegressionClassifier(vectorizer, model)

def save_distilbert_checkpoint(path, model, optimizer, epoch, best_dev_acc):
    """
    To save a DistilBERT checkpoint given path, model, optimizer, epoch, and best_dev_acc.
    """
    ensure_dir(os.path.dirname(path))
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_dev_acc": best_dev_acc},
        path)

def load_distilbert_checkpoint(path, model, optimizer=None):
    """
    To load a DistilBERT checkpoint given path, model, and optimizer.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, next(model.parameters()).device)
    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_dev_acc": checkpoint.get("best_dev_acc", -1.0),
    }

def append_epoch_metrics(path, epoch, train_loss, dev_metrics):
    """
    To append epoch metrics given path, epoch, train_loss, and dev_metrics.
    """
    ensure_dir(os.path.dirname(path))
    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "dev_accuracy": dev_metrics["accuracy"],
        "dev_macro_f1": dev_metrics["macro_f1"],
        "dev_confusion_matrix": dev_metrics["confusion_matrix"].tolist(),
    }
    with open(path, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record) + "\n")

def save_run_config(path, args):
    """
    To save run config given path and args.
    """
    ensure_dir(os.path.dirname(path))
    config = {
        "transformer_model_name": args.transformer_model_name,
        "transformer_lr": args.transformer_lr,
        "transformer_batch_size": args.transformer_batch_size,
        "max_length": args.max_length,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "processed_train_path": getattr(args, "processed_train_path", None),
        "processed_dev_path": getattr(args, "processed_dev_path", None),
        "resume_from_checkpoint": getattr(args, "resume_from_checkpoint", None),
    }
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(config, outfile, indent=2)

def train_distilbert_classifier(args, train_exs, dev_exs):
    """
    To train a DistilBERT classifier given args, train_exs, and dev_exs.
    """
    set_random_seeds(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.transformer_model_name,
        num_labels=3,
    )
    device = _get_torch_device()
    model = model.to(device)
    transformer_batch_size = getattr(args, "transformer_batch_size", args.batch_size)
    output_dir = getattr(args, "output_dir", getattr(args, "distilbert_output_dir", "artifacts/distilbert/run_001"))
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    args.output_dir = output_dir
    train_dataloader = _build_transformer_dataloader(
        train_exs,
        tokenizer,
        args.max_length,
        transformer_batch_size,
        True,
    )
    dev_dataloader = _build_transformer_dataloader(
        dev_exs,
        tokenizer,
        args.max_length,
        transformer_batch_size,
        False,
    )
    learning_rate = getattr(args, "transformer_lr", args.lr)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    ensure_dir(output_dir)
    save_run_config(os.path.join(output_dir, "config.json"), args)
    start_epoch = 0
    best_dev_acc = -1.0

    # Resume both model and optimizer state so training can continue cleanly.
    if resume_from_checkpoint is not None:
        checkpoint_metadata = load_distilbert_checkpoint(resume_from_checkpoint, model, optimizer)
        start_epoch = checkpoint_metadata["epoch"]
        best_dev_acc = checkpoint_metadata["best_dev_acc"]
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    for epoch in range(start_epoch, args.num_epochs):
        total_loss = _train_distilbert_epoch(model, train_dataloader, optimizer, device)
        avg_train_loss = total_loss / len(train_exs)
        dev_metrics = _evaluate_distilbert(model, dev_dataloader, device)
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        current_best_dev_acc = best_dev_acc
        if dev_metrics["accuracy"] > current_best_dev_acc:
            current_best_dev_acc = dev_metrics["accuracy"]
        append_epoch_metrics(
            os.path.join(output_dir, "metrics.jsonl"),
            epoch + 1,
            avg_train_loss,
            dev_metrics,
        )
        save_distilbert_checkpoint(checkpoint_path, model, optimizer, epoch + 1, current_best_dev_acc)
        # Keep a separate best checkpoint for the simplest evaluation-time restore path.
        if dev_metrics["accuracy"] > best_dev_acc:
            best_dev_acc = current_best_dev_acc
            save_distilbert_checkpoint(
                os.path.join(output_dir, "best_model.pt"),
                model,
                optimizer,
                epoch + 1,
                best_dev_acc,
            )
        print(f"Epoch {epoch + 1}")
        print(f"train_loss={avg_train_loss:.4f}")
        print(f"dev_acc={dev_metrics['accuracy']:.4f} dev_macro_f1={dev_metrics['macro_f1']:.4f}")
        print("------------------------------")

    return DistilBertClaimClassifier(model, tokenizer, device, args.max_length)

def _build_tfidf_vectorizer():
    """
    To build a TF-IDF vectorizer given no arguments.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

def train_deep_averaging_network(args, train_exs, dev_exs):
    """
    To train a deep averaging network given args, train_exs, and dev_exs.
    """
    set_random_seeds(args.seed)
    if args.word_vecs_path is not None:
        word_embeddings = read_word_embeddings(args.word_vecs_path)
        indexer = word_embeddings.word_indexer
        embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True, padding_idx=0)
    else:
        indexer = _build_word_indexer(train_exs)
        embedding_layer = nn.Embedding(len(indexer), args.embedding_dim, padding_idx=0)

    train_data = _build_train_data(train_exs, indexer)
    device = _get_torch_device()
    model = DeepAveragingNetwork(embedding_layer, args.hidden_size, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    classifier = NeuralClaimClassifier(model, indexer, device, args.batch_size)

    for epoch in range(args.num_epochs):
        _train_dan_epoch(model, train_data, optimizer, loss_function, device, args.batch_size)

    return classifier

def _tokenize_transformer_batch(tokenizer, claims, evidence_texts, max_length):
    """
    To tokenize transformer inputs given tokenizer, claims, evidence_texts, and max_length.
    """
    return tokenizer(
        claims,
        evidence_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

def _build_transformer_dataloader(exs, tokenizer, max_length, batch_size, shuffle):
    """
    To build a transformer dataloader given exs, tokenizer, max_length, batch_size, and shuffle.
    """
    if not exs:
        return DataLoader([], batch_size=batch_size, shuffle=False)

    claims = [ex.claim for ex in exs]
    evidence_texts = [ex.evidence_text for ex in exs]
    # Tokenize the whole split once, then slice the tensors back into example features.
    encodings = _tokenize_transformer_batch(tokenizer, claims, evidence_texts, max_length)
    features = []

    for ex_idx, ex in enumerate(exs):
        feature = {key: value[ex_idx] for key, value in encodings.items()}
        if ex.label is not None:
            feature["labels"] = torch.tensor(ex.label, dtype=torch.long)
        features.append(feature)

    return DataLoader(features, batch_size=batch_size, shuffle=shuffle)

def _train_distilbert_epoch(model, dataloader, optimizer, device):
    """
    To train DistilBERT for one epoch given model, dataloader, optimizer, and device.
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        batch = _move_transformer_batch_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["labels"].size(0)

    return total_loss

def _evaluate_distilbert(model, dataloader, device):
    """
    To evaluate DistilBERT given model, dataloader, and device.
    """
    model.eval()
    golds = []
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_transformer_batch_to_device(batch, device)
            golds.extend(batch["labels"].cpu().tolist())
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    return compute_metrics(golds, preds)

def _build_word_indexer(train_exs):
    """
    To build a word indexer given train_exs.
    """
    indexer = Indexer()

    for ex in train_exs:
        for word in ex.words:
            indexer.add_and_get_index(word)

    return indexer

def _words_to_indices(words, indexer):
    """
    To map words to indices given words and indexer.
    """
    indices = []
    for word in words:
        indices.append(indexer.index_of(word))
    return indices if indices else [indexer.index_of("UNK")]

def _deconstruct_batch(batch):
    """
    To separate batch inputs and labels given batch.
    """
    batch_word_indices = []
    batch_labels = []

    for word_indices, label in batch:
        batch_word_indices.append(word_indices)
        batch_labels.append(label)

    return batch_word_indices, batch_labels

def _build_train_data(train_exs, indexer):
    """
    To build indexed training data given train_exs and indexer.
    """
    train_data = []
    for ex in train_exs:
        train_data.append((_words_to_indices(ex.words, indexer), ex.label))
    return train_data

def _train_dan_epoch(model, train_data, optimizer, loss_function, device, batch_size):
    """
    To train the DAN for one epoch given model, train_data, optimizer, loss_function, device, and batch_size.
    """
    random.shuffle(train_data)
    total_loss = 0.0

    for start_idx in range(0, len(train_data), batch_size):
        batch = train_data[start_idx:start_idx + batch_size]
        batch_word_indices, batch_labels = _deconstruct_batch(batch)
        word_tensor = form_input(pad_batch(batch_word_indices)).to(device)
        label_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(word_tensor)
        loss = loss_function(logits, label_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch)

    return total_loss

def _move_transformer_batch_to_device(batch, device):
    """
    To move a transformer batch to a device given batch and device.
    """
    moved_batch = {}
    for key, value in batch.items():
        moved_batch[key] = value.to(device)
    return moved_batch

def _move_optimizer_state_to_device(optimizer, device):
    """
    To move optimizer state to a device given optimizer and device.
    """
    # Optimizer tensors are loaded on CPU by default, so move them with the model.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

def _get_torch_device():
    """
    To get a torch device given no arguments.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
