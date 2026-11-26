"""Load Andrew's PyTorch FFNN bundle (models/ffnn_complete.joblib) so it behaves like
our scikit-learn pipelines: it takes raw text and has predict_proba / predict / classes_.

The joblib file is a dict saved from the training notebook:
    tfidf               fitted TfidfVectorizer (50,000 features, unigrams + bigrams)
    model_state_dict    PyTorch weights of `network` (an nn.Sequential)
    input_dim           50000
    hidden_dims         [64, 32]
    dropout             [0.3, 0.2]
    classification_threshold  0.5
    class_mapping       {0: "negative", 1: "positive"}

Architecture rebuilt here (must match the training code):
    Linear(50000 -> 64) -> ReLU -> Dropout(0.3)
    Linear(64 -> 32)    -> ReLU -> Dropout(0.2)
    Linear(32 -> 1)     -> sigmoid gives P(positive)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def load_model(path: Path):
    """Load any model file in models/. Plain sklearn pipelines are returned as is;
    an FFNN bundle (a dict with `model_state_dict`) is wrapped in FFNNClassifier."""
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return FFNNClassifier(obj)
    return obj


class FFNNClassifier:
    def __init__(self, bundle: dict):
        import torch
        from torch import nn

        self._torch = torch
        self.vectorizer = bundle["tfidf"]
        self.threshold = float(bundle.get("classification_threshold", 0.5))
        mapping = bundle.get("class_mapping", {0: "negative", 1: "positive"})
        self.classes_ = np.array([mapping[0], mapping[1]])

        layers, in_dim = [], bundle["input_dim"]
        for units, p in zip(bundle["hidden_dims"], bundle["dropout"]):
            layers += [nn.Linear(in_dim, units), nn.ReLU(), nn.Dropout(p)]
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        self.net = Net()
        self.net.load_state_dict(bundle["model_state_dict"])
        self.net.eval()  # turns dropout off for prediction

    def predict_proba(self, texts, batch_size: int = 2048):
        """Returns an (n, 2) array: columns follow classes_ (negative, positive)."""
        texts = list(texts)
        p_pos = []
        for i in range(0, len(texts), batch_size):
            x = self.vectorizer.transform(texts[i:i + batch_size]).toarray()
            with self._torch.no_grad():
                logits = self.net(self._torch.tensor(x, dtype=self._torch.float32))
            p_pos.append(self._torch.sigmoid(logits).squeeze(1).numpy())
        p_pos = np.concatenate(p_pos) if p_pos else np.zeros(0)
        return np.column_stack([1 - p_pos, p_pos])

    def predict(self, texts):
        p_pos = self.predict_proba(texts)[:, 1]
        return np.where(p_pos >= self.threshold, self.classes_[1], self.classes_[0])
