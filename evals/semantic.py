"""Rubric-based semantic evaluation (fixture backend).

Two backends:

- ``fixture`` (default, offline): scores a draft answer
  against human-authored substantive markers per topic.
  A topic is ``adequate`` only when a marker appears in
  a sentence WITHOUT an unavailability disclaimer — the
  same disclaimer detection the deterministic checker
  uses, so "conditional routing details were not
  available" never scores as an explanation. This scores
  FIXTURE answers; it is not a claim of factual
  correctness.
- ``judge`` (opt-in, ``EVAL_LLM_JUDGE=1``): an
  LLM-as-judge rubric scorer. DISABLED by default and
  never used by tests — it requires a configured live
  provider.

The two are deliberately separate from routing
enforcement: semantic scores are diagnostics; only the
deterministic gates (citations, coverage, official
retrieval) influence the graph's routing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from agent.completeness import _substantive_sentences

__all__ = [
    "SemanticResult",
    "TopicScore",
    "evaluate_semantics",
    "rubric_judge_enabled",
]


@dataclass(frozen=True)
class TopicScore:
    """One topic's rubric score."""

    topic: str
    adequate: bool
    matched_markers: tuple[str, ...]
    matched_in_disclaimer: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "adequate": self.adequate,
            "matched_markers": list(self.matched_markers),
            "matched_in_disclaimer": list(
                self.matched_in_disclaimer
            ),
        }


@dataclass(frozen=True)
class SemanticResult:
    """Semantic evaluation of one draft answer."""

    case_id: str
    topics: tuple[TopicScore, ...] = ()
    overall_adequate: bool = True
    reviewer_approved: bool = False
    reviewer_false_approval: bool = False
    judge_backend: str = "fixture"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def inadequate_topics(self) -> tuple[str, ...]:
        return tuple(
            score.topic
            for score in self.topics
            if not score.adequate
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "backend": self.judge_backend,
            "topics": [score.to_dict() for score in self.topics],
            "overall_adequate": self.overall_adequate,
            "reviewer_approved": self.reviewer_approved,
            "reviewer_false_approval": self.reviewer_false_approval,
            "inadequate_topics": list(self.inadequate_topics),
            "notes": list(self.notes),
        }


def _marker_in_sentences(
    marker: str, sentences: list[str]
) -> bool:
    import re

    pattern = re.compile(re.escape(marker), re.IGNORECASE)
    return any(pattern.search(sentence) for sentence in sentences)


def score_topic(
    answer: str, topic: Any
) -> TopicScore:
    """Score one topic against its rubric.

    A marker counts only when it appears in a sentence
    that is NOT an unavailability disclaimer; markers that
    appear only inside disclaimers are recorded
    separately (they prove the topic was MENTIONED, not
    explained).
    """
    substantive = _substantive_sentences(answer)
    matched: list[str] = []
    disclaimed: list[str] = []
    for marker in topic.expected_markers:
        if _marker_in_sentences(marker, substantive):
            matched.append(marker)
        elif _marker_in_sentences(marker, [answer]):
            disclaimed.append(marker)
    return TopicScore(
        topic=topic.topic,
        adequate=bool(matched),
        matched_markers=tuple(matched),
        matched_in_disclaimer=tuple(disclaimed),
    )


def evaluate_semantics(
    case: Any,
    answer: str,
    *,
    reviewer_approved: bool = False,
) -> SemanticResult:
    """Fixture-based semantic evaluation of a draft."""
    topics = tuple(
        score_topic(answer, rubric) for rubric in case.rubric
    )
    overall = all(score.adequate for score in topics)
    false_approval = reviewer_approved and not overall
    return SemanticResult(
        case_id=case.case_id,
        topics=topics,
        overall_adequate=overall,
        reviewer_approved=reviewer_approved,
        reviewer_false_approval=false_approval,
        judge_backend="fixture",
        notes=(
            (
                "reviewer APPROVED a semantically "
                "inadequate answer",
            )
            if false_approval
            else ()
        ),
    )


def rubric_judge_enabled() -> bool:
    """LLM-as-judge is opt-in and off by default."""
    return os.environ.get("EVAL_LLM_JUDGE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


class RubricJudge:
    """Optional LLM-as-judge scorer (opt-in).

    NOT constructed unless ``EVAL_LLM_JUDGE`` is set —
    the offline harness never instantiates it, so no
    provider calls happen during tests or default
    benchmark runs. Scores all five semantic dimensions
    per the Phase 4A brief: topic adequacy, citation
    faithfulness, factual correctness relative to the
    supplied evidence, completeness and reviewer
    approval correctness.
    """

    def __init__(self, model: Any) -> None:
        if not rubric_judge_enabled():
            raise RuntimeError(
                "RubricJudge requires EVAL_LLM_JUDGE=1 "
                "(live provider call)"
            )
        self._model = model

    def score(self, case: Any, answer: str) -> dict[str, Any]:
        """Judge the answer against the case rubric."""
        rubric_text = "\n".join(
            f"- {rubric.topic}: substantive explanation "
            f"must cover {'/'.join(rubric.expected_markers)}"
            for rubric in case.rubric
        )
        prompt = (
            "You are a strict evaluator. Score the answer "
            "against each rubric topic (is the MECHANISM "
            "explained, not merely named?). Also state "
            "whether cited evidence supports the claims "
            "and whether the answer is complete. Reply as "
            "JSON: "
            '{"topics": {"<topic>": "adequate|inadequate"}, '
            '"overall_adequate": true|false, '
            '"citation_faithfulness": "pass|fail", '
            '"factual_correctness": "pass|fail", '
            '"completeness": "pass|fail"}\n\n'
            f"RUBRIC:\n{rubric_text}\n\nANSWER:\n{answer}"
        )
        import json

        from langchain_core.messages import HumanMessage

        response = self._model.invoke(
            [HumanMessage(content=prompt)]
        )
        text = str(response.content)
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {
                "error": "judge output was not valid JSON",
                "topics": {},
                "overall_adequate": False,
            }
