"""The stub adapter.

Unlike metrics/, there is no independently correct answer here -- any deterministic
distribution would do, so these cannot be hand-computed from first principles. Their
job is to detect *change*: that the stub still returns the same numbers it did
yesterday, and that its two methods still agree with each other. Everything the stub
is useful for depends on those two properties.
"""

import pytest

from madcal.models import StubAdapter

CHOICES = ("A", "B", "C", "D")


@pytest.fixture
def stub() -> StubAdapter:
    return StubAdapter()


def test_distribution_is_stable_across_processes(stub):
    """The hash() regression guard -- the single most important test in this file.

    Python randomises string hashing per process, so if `_rng` ever goes back to
    the builtin hash(), every number the stub produces changes on every run. That
    surfaces as tests failing intermittently in CI, which reads as flakiness rather
    than a bug, and it silently invalidates any stored stub results.

    Hardcoding the vector catches it, because these values were produced in a
    different process from the one running the test.
    """
    result = stub.score_choices(["q1"], [CHOICES])[0].probabilities()
    assert result == pytest.approx(
        (0.0839266875301645, 0.21588108318242294, 0.3405502535899185, 0.3596419756974942)
    )


def test_score_and_generate_agree_on_the_argmax(stub):
    """Both methods must share `_distribution`.

    If they ever compute their own numbers, the stub would report one confidence
    and answer with a different option -- confidences attached to the wrong answer,
    in the one component whose entire job is being predictable. Nothing would raise.
    """
    prompts = ["q1", "q2", "q3"]
    scores = stub.score_choices(prompts, [CHOICES] * len(prompts))
    generations = stub.generate(prompts, n=1, temperature=0.0)

    for prompt, score, samples in zip(prompts, scores, generations, strict=True):
        probs = score.probabilities()
        favoured = CHOICES[probs.index(max(probs))]
        text = samples[0].text
        assert favoured in text, (
            f"{prompt}: score_choices favours {favoured} "
            f"(probs={[round(p, 3) for p in probs]}) but generate returned {text!r}"
        )


def test_greedy_repeats_and_sampling_varies(stub):
    """Pins the semantics signal 3 depends on.

    Semantic-cluster tightness measures disagreement across N resamples. If
    temperature were ignored and every sample came back identical, that signal
    would read as perfect agreement on every question -- a plausible-looking
    number that means nothing.
    """
    # Determinism is checked with n=1 called repeatedly, not with n>1 at temperature 0.
    # TransformersAdapter rejects that combination outright -- n greedy samples are n
    # copies of one answer, which would hand signal 3 fabricated agreement -- so the
    # stub is not exercised down a path the real adapter refuses.
    greedy = [stub.generate(["q1"], n=1, temperature=0.0)[0][0].text for _ in range(3)]
    assert greedy == [greedy[0]] * 3, f"greedy decoding should repeat, got {greedy}"

    sampled = {g.text for g in stub.generate(["q1"], n=20, temperature=1.0)[0]}
    assert len(sampled) > 1, f"temperature=1.0 produced no variation: {sampled}"

    def texts(seed: int) -> list[str]:
        return [g.text for g in stub.generate(["q1"], n=5, temperature=1.0, seed=seed)[0]]

    assert texts(7) == texts(7), "same seed must reproduce the same samples"
    assert texts(7) != texts(8), f"seeds 7 and 8 produced identical samples: {texts(7)}"


@pytest.mark.parametrize("prompt", ["q1", "q2", "q3"])
def test_option_mass_is_below_one(stub, prompt):
    """`_distribution` scales by a random mass so this path is exercised.

    A real model leaks probability onto the rest of its vocabulary. If the stub
    always summed to exactly 1, `option_mass()` would return 1.0 in every test and
    the signal would look constant -- hiding a bug in whatever consumes it.
    """
    score = stub.score_choices([prompt], [CHOICES])[0]
    assert sum(score.probabilities()) == pytest.approx(1.0)

    mass = score.option_mass()
    assert 0.0 < mass < 1.0, f"{prompt}: option_mass {mass} is outside (0, 1)"


@pytest.mark.parametrize(
    "kwargs",
    [{"n": 0}, {"n": -1}, {"temperature": -0.1}],
    ids=["n-zero", "n-negative", "temperature-negative"],
)
def test_generate_rejects_invalid_arguments(stub, kwargs):
    """n=0 would return empty sample lists, which reads downstream as "the model
    produced nothing" rather than "you asked for nothing"."""
    with pytest.raises(ValueError):
        stub.generate(["q"], **kwargs)


@pytest.mark.parametrize(
    ("prompts", "choices"),
    [(["a", "b"], [CHOICES]), (["a"], [CHOICES, CHOICES])],
    ids=["more-prompts-than-choices", "more-choices-than-prompts"],
)
def test_score_choices_rejects_misaligned_input(stub, prompts, choices):
    """Misalignment would silently score the wrong options against the wrong prompt."""
    with pytest.raises(ValueError):
        stub.score_choices(prompts, choices)


def test_confidence_placeholder_is_the_renormalised_probability_of_the_answer(stub):
    templated = StubAdapter(response_template="Answer: {letter} Confidence: {confidence}%")
    probabilities = stub.score_choices(["q1"], [CHOICES])[0].probabilities()
    text = templated.generate(["q1"], temperature=0.0)[0][0].text
    assert text == f"Answer: D Confidence: {round(100 * max(probabilities))}%"
    assert text == "Answer: D Confidence: 36%"
