"""Natural-language-inference judge deciding whether one text entails another."""

from collections.abc import Sequence

from madcal.models.transformers import cached_commit_sha


class NLIJudge:
    """Entailment classifier: a pair entails when its most likely label is entailment."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_id = model_id
        self.revision = cached_commit_sha(model_id, revision)
        self.batch_size = batch_size
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=self.revision, local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=self.revision,
            dtype=torch.float32,
            device_map=device,
            local_files_only=True,
        ).eval()
        labels = {str(name).lower(): index for index, name in self._model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"{model_id!r} has no entailment label: {sorted(labels)}")
        self._entailment = labels["entailment"]

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[bool]:
        verdicts: list[bool] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            encoded = self._tokenizer(
                [premise for premise, _ in chunk],
                [hypothesis for _, hypothesis in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._model.device)
            with self._torch.no_grad():
                predicted = self._model(**encoded).logits.argmax(dim=-1)
            verdicts.extend(bool(label == self._entailment) for label in predicted.tolist())
        return verdicts
