from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from madcal.models import NLIJudge

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
PAIRS = [
    ("The capital of France is Paris.", "The capital of France is Paris."),
    ("The capital of France is Paris.", "The capital of France is Lyon."),
    ("A man is playing a guitar on stage.", "A man is performing music."),
    ("A man is performing music.", "A man is playing a guitar on stage."),
    ("A dog runs in the park.", "A cat sleeps on the sofa."),
]


def _is_cached() -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(NLI_MODEL, local_files_only=True)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _is_cached(), reason=f"{NLI_MODEL} is not cached")


@pytest.fixture(scope="module")
def judge() -> NLIJudge:
    return NLIJudge(NLI_MODEL, device="cpu", batch_size=2)


def test_verdicts_match_an_independent_forward_pass(judge):
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, local_files_only=True)
    encoded = tokenizer(
        [premise for premise, _ in PAIRS],
        [hypothesis for _, hypothesis in PAIRS],
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        labels = model.eval()(**encoded).logits.argmax(dim=-1).tolist()
    entailment = {name.lower(): index for index, name in model.config.id2label.items()}[
        "entailment"
    ]
    assert judge(PAIRS) == [label == entailment for label in labels]


def test_a_sentence_entails_itself_and_not_a_contradiction(judge):
    assert judge(PAIRS[:2]) == [True, False]


def test_batching_does_not_change_verdicts(judge):
    single = NLIJudge(NLI_MODEL, device="cpu", batch_size=1)
    assert judge(PAIRS) == single(PAIRS)


def test_the_revision_is_a_commit_sha(judge):
    assert judge.revision is not None and len(judge.revision) == 40


def test_a_sentencepiece_only_tokenizer_loads_and_matches_sentencepiece(tmp_path):
    import shutil

    import sentencepiece
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(NLI_MODEL, local_files_only=True))
    for name in ("config.json", "spm.model"):
        shutil.copy(snapshot / name, tmp_path / name)
    (tmp_path / "tokenizer_config.json").write_text('{"do_lower_case": false, "vocab_type": "spm"}')
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    reference = sentencepiece.SentencePieceProcessor(model_file=str(tmp_path / "spm.model"))
    for text in ("Who wrote Hamlet? William Shakespeare", "£340m Aviva Stadium, naïve café"):
        assert tokenizer.tokenize(text) == reference.encode(text, out_type=str)
