"""Gemini model capability + free-tier eligibility registry (4E.3).

Two SEPARATE sources of truth, deliberately never conflated:

1. **Capabilities** (what a model accepts): which thinking control the
   model family uses — ``thinking_level`` (Gemini 3+) or
   ``thinking_budget`` (Gemini 2.5) — and which levels that SPECIFIC
   model supports. Taken from the official Gemini thinking docs
   (``ai.google.dev/gemini-api/docs/gemini`` and
   ``.../generate-content/thinking``): e.g. ``gemini-3.8-flash``
   rejects ``minimal`` with 400 INVALID_ARGUMENT, while
   ``gemini-3.6-flash`` accepts it.

2. **Free-tier eligibility** (what the user may be charged): maintained
   here by hand from the official pricing page
   (``ai.google.dev/gemini-api/docs/pricing``), verified on
   ``VERIFIED_ON``. The ``models.list`` API does NOT prove eligibility
   — an API listing only proves the model exists for this key — so a
   live discovery result can annotate availability but can NEVER make
   a model selectable. Unknown eligibility is excluded from the
   free-only selector (never assumed free, never assumed paid).

Registry entries with ``status`` ``deprecated``/``shutdown`` are
TOMBSTONES: they stay listed so a historical run that selected them
fails fast with a clear ``model_unavailable`` error instead of a
provider round-trip — and so nothing ever silently substitutes a
different model.

No network call happens in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "GEMINI_MODEL_REGISTRY",
    "SELECTABLE_STATUSES",
    "TOMBSTONE_STATUSES",
    "THINKING_LEVELS",
    "ModelSelectionError",
    "ModelSpec",
    "ModelUnavailableError",
    "RunModelConfig",
    "control_by_family",
    "family_of",
    "is_selectable",
    "resolve_run_model_config",
    "selectable_models",
    "spec_for",
    "thinking_kwargs",
]

#: The full selector vocabulary, in UI order. ``auto`` means "model
#: default" and is expressed on the wire by SENDING NOTHING — it is
#: never a literal API value (the API has no "auto" token).
THINKING_LEVELS: tuple[str, ...] = (
    "auto",
    "minimal",
    "low",
    "medium",
    "high",
)

#: Documented per-family budgets for budget-controlled (Gemini 2.5)
#: models, from the official OpenAI-compatibility mapping table:
#: minimal -> 1,024, low -> 1,024, medium -> 8,192, high -> 24,576.
#: ``auto`` is absent by design: it means "omit thinking_budget and
#: let the model decide" (2.5 dynamic thinking).
BUDGET_BY_LEVEL: dict[str, int] = {
    "minimal": 1024,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}

StatusLiteral = Literal["stable", "preview", "deprecated", "shutdown"]
FreeTierLiteral = Literal["eligible", "paid_only", "unknown"]
ThinkingControl = Literal["level", "budget", "none"]

#: Statuses a run may be created with. ``deprecated``/``shutdown``
#: entries exist only to reject new selections and to fail historical
#: ones clearly.
SELECTABLE_STATUSES: frozenset[str] = frozenset({"stable", "preview"})
TOMBSTONE_STATUSES: frozenset[str] = frozenset({"deprecated", "shutdown"})

#: Date the free-tier/capability entries below were verified against
#: the official pricing and thinking documentation.
VERIFIED_ON = "2026-10-14"

_DOCS_BASE = "https://ai.google.dev/gemini-api/docs/models/"
_THINKING_DOC = "https://ai.google.dev/gemini-api/docs/gemini#thinking"


@dataclass(frozen=True)
class ModelSpec:
    """Everything the backend will admit about one Gemini model."""

    model_id: str
    display_name: str
    family: str  # "gemini-3" | "gemini-2.5" | ...
    status: StatusLiteral
    free_tier: FreeTierLiteral
    thinking_control: ThinkingControl
    #: Supported values from THINKING_LEVELS (``auto`` is always
    #: implicitly supported — it sends nothing — but is listed here
    #: so the UI can render one honest list per model).
    thinking_levels: tuple[str, ...]
    #: What the model does when no thinking parameter is sent.
    default_thinking: str
    docs_url: str
    verified_on: str = VERIFIED_ON
    notes: str = ""
    #: Official deprecation/shutdown date for tombstones.
    retired_on: str = ""

    @property
    def free_tier_eligible(self) -> bool:
        return (
            self.free_tier == "eligible"
            and self.status in SELECTABLE_STATUSES
        )


def _spec(**kwargs: object) -> ModelSpec:
    return ModelSpec(**kwargs)  # type: ignore[arg-type]


#: Backend-managed catalogue. Order matters: it is the display order
#: of ``GET /api/models`` (default model first).
GEMINI_MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.model_id: spec
    for spec in (
        # -- free-tier eligible, text chat (selectable) -------------
        _spec(
            model_id="gemini-3.8-flash",
            display_name="Gemini 3.8 Flash",
            family="gemini-3",
            status="stable",
            free_tier="eligible",
            thinking_control="level",
            thinking_levels=("auto", "low", "medium", "high"),
            default_thinking="medium",
            docs_url=_DOCS_BASE + "gemini-3.8-flash",
            notes=(
                "Default model. 'minimal' is NOT supported on this "
                "model (the API rejects it); reasoning defaults to "
                "medium. Free tier is quota-limited."
            ),
        ),
        _spec(
            model_id="gemini-3.6-flash",
            display_name="Gemini 3.6 Flash",
            family="gemini-3",
            status="stable",
            free_tier="eligible",
            thinking_control="level",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="medium",
            docs_url=_DOCS_BASE + "gemini-3.6-flash",
            notes="Previous-generation Flash. Free tier is quota-limited.",
        ),
        _spec(
            model_id="gemini-3.5-flash-lite",
            display_name="Gemini 3.5 Flash-Lite",
            family="gemini-3",
            status="stable",
            free_tier="eligible",
            thinking_control="level",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="minimal",
            docs_url=_DOCS_BASE + "gemini-3.5-flash-lite",
            notes=(
                "Cheapest eligible option; defaults to minimal "
                "thinking. Free tier is quota-limited."
            ),
        ),
        _spec(
            model_id="gemini-3.1-flash-lite",
            display_name="Gemini 3.1 Flash-Lite",
            family="gemini-3",
            status="stable",
            free_tier="eligible",
            thinking_control="level",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="minimal",
            docs_url=_DOCS_BASE + "gemini-3.1-flash-lite",
            notes="Fast, low-cost. Free tier is quota-limited.",
        ),
        _spec(
            model_id="gemini-3-flash-preview",
            display_name="Gemini 3 Flash (preview)",
            family="gemini-3",
            status="preview",
            free_tier="eligible",
            thinking_control="level",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="high",
            docs_url=_DOCS_BASE + "gemini-3-flash-preview",
            notes=(
                "PREVIEW: may change or be retired without notice and "
                "has more restrictive rate limits. Free tier is "
                "quota-limited."
            ),
        ),
        _spec(
            model_id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            family="gemini-2.5",
            status="stable",
            free_tier="eligible",
            thinking_control="budget",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="auto",
            docs_url=_DOCS_BASE + "gemini-2.5-pro",
            notes=(
                "Budget-controlled family: levels map to token budgets "
                "(minimal/low 1,024; medium 8,192; high 24,576). "
                "Thinking cannot be fully disabled on Pro. Free tier "
                "is quota-limited."
            ),
        ),
        _spec(
            model_id="gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            family="gemini-2.5",
            status="stable",
            free_tier="eligible",
            thinking_control="budget",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="auto",
            docs_url=_DOCS_BASE + "gemini-2.5-flash",
            notes=(
                "Budget-controlled family: levels map to token budgets "
                "(minimal/low 1,024; medium 8,192; high 24,576). "
                "Free tier is quota-limited."
            ),
        ),
        _spec(
            model_id="gemini-2.5-flash-lite",
            display_name="Gemini 2.5 Flash-Lite",
            family="gemini-2.5",
            status="stable",
            free_tier="eligible",
            thinking_control="budget",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="auto",
            docs_url=_DOCS_BASE + "gemini-2.5-flash-lite",
            notes=(
                "Budget-controlled family; the model does not think "
                "unless a budget is set. Free tier is quota-limited."
            ),
        ),
        # -- NOT free-tier eligible (reported, never selectable) -----
        _spec(
            model_id="gemini-4-argon",
            display_name="Gemini 4 Argon",
            family="gemini-4",
            status="stable",
            free_tier="paid_only",
            thinking_control="level",
            thinking_levels=("auto", "low", "medium", "high"),
            default_thinking="medium",
            docs_url=_DOCS_BASE + "gemini-4-argon",
            notes=(
                "Paid tier only — explicitly not available on the "
                "Free tier. Excluded from the free-only selector."
            ),
        ),
        _spec(
            model_id="gemini-3.1-pro-preview",
            display_name="Gemini 3.1 Pro (preview)",
            family="gemini-3",
            status="preview",
            free_tier="paid_only",
            thinking_control="level",
            thinking_levels=("auto", "low", "medium", "high"),
            default_thinking="high",
            docs_url=_DOCS_BASE + "gemini-3.1-pro-preview",
            notes=(
                "No free-tier entry in the official pricing table "
                "(paid only). Excluded from the free-only selector."
            ),
        ),
        _spec(
            model_id="gemini-3.7-flash",
            display_name="Gemini 3.7 Flash",
            family="gemini-3",
            status="stable",
            free_tier="unknown",
            thinking_control="level",
            thinking_levels=("auto", "low", "medium", "high"),
            default_thinking="medium",
            docs_url=_DOCS_BASE + "gemini-3.7-flash",
            notes=(
                "Free-tier eligibility NOT confirmed in the official "
                "pricing table at the verification date — excluded "
                "from the selector until it is (never assumed free)."
            ),
        ),
        _spec(
            model_id="gemini-3.5-flash",
            display_name="Gemini 3.5 Flash",
            family="gemini-3",
            status="stable",
            free_tier="unknown",
            thinking_control="level",
            thinking_levels=("auto", "minimal", "low", "medium", "high"),
            default_thinking="medium",
            docs_url=_DOCS_BASE + "gemini-3.5-flash",
            notes=(
                "Free-tier eligibility NOT confirmed in the official "
                "pricing table at the verification date — excluded "
                "from the selector until it is (never assumed free)."
            ),
        ),
        # -- tombstones (historical selections fail fast) -----------
        _spec(
            model_id="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash (retired)",
            family="gemini-2",
            status="shutdown",
            free_tier="unknown",
            thinking_control="none",
            thinking_levels=("auto",),
            default_thinking="auto",
            docs_url=_DOCS_BASE + "gemini-2.0-flash",
            retired_on="2026-06-01",
            notes=(
                "Deprecated and shut down on 2026-06-01. Kept only so "
                "old runs fail with a clear error instead of silently "
                "using a different model."
            ),
        ),
        _spec(
            model_id="gemini-2.0-flash-lite",
            display_name="Gemini 2.0 Flash-Lite (retired)",
            family="gemini-2",
            status="shutdown",
            free_tier="unknown",
            thinking_control="none",
            thinking_levels=("auto",),
            default_thinking="auto",
            docs_url=_DOCS_BASE + "gemini-2.0-flash-lite",
            retired_on="2026-06-01",
            notes=(
                "Deprecated and shut down on 2026-06-01. Kept only so "
                "old runs fail with a clear error instead of silently "
                "using a different model."
            ),
        ),
    )
}


class ModelSelectionError(ValueError):
    """A requested model/thinking combination is not acceptable.

    ``category`` is the machine-readable API error category (never
    raw provider text): ``unknown_model``, ``model_not_selectable``
    or ``unsupported_thinking_level`` / ``invalid_configuration``.
    """

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


class ModelUnavailableError(RuntimeError):
    """A run's persisted model is no longer usable.

    Raised at EXECUTION time (fresh run or resume) when the frozen
    configuration names a retired model. There is deliberately NO
    fallback: the run fails with ``model_unavailable`` rather than
    silently resuming on a different model.
    """


@dataclass(frozen=True)
class RunModelConfig:
    """Immutable, validated per-run model configuration."""

    model_id: str
    thinking_level: str  # one of THINKING_LEVELS

    def as_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id, "thinking_level": self.thinking_level}


def spec_for(model_id: object) -> ModelSpec | None:
    """Registry entry for an exact model id (``None`` when unknown)."""
    if not isinstance(model_id, str):
        return None
    return GEMINI_MODEL_REGISTRY.get(model_id.removeprefix("models/"))


def family_of(model_id: str) -> str:
    """Best-effort family for IDs outside the registry (custom config)."""
    name = model_id.removeprefix("models/")
    for prefix in ("gemini-4", "gemini-3", "gemini-2.5", "gemini-2.0"):
        if name.startswith(prefix):
            return prefix
    return "unknown"


def is_selectable(spec: ModelSpec) -> bool:
    """Free-only selector rule: eligible tier AND not retired."""
    return spec.free_tier == "eligible" and spec.status in SELECTABLE_STATUSES


def selectable_models() -> list[ModelSpec]:
    return [s for s in GEMINI_MODEL_REGISTRY.values() if is_selectable(s)]


def _validate_thinking(spec: ModelSpec, thinking_level: str) -> None:
    if thinking_level not in THINKING_LEVELS:
        raise ModelSelectionError(
            "unsupported_thinking_level",
            f"thinking_level must be one of {', '.join(THINKING_LEVELS)}",
        )
    if thinking_level == "auto":
        return  # always valid: sends no thinking parameter
    if thinking_level not in spec.thinking_levels:
        supported = ", ".join(spec.thinking_levels)
        raise ModelSelectionError(
            "unsupported_thinking_level",
            f"'{thinking_level}' is not supported by {spec.display_name} "
            f"(supported: {supported})",
        )


def resolve_run_model_config(
    model_id: object,
    thinking_level: object,
    *,
    default_model_id: str,
    default_thinking_level: str | None = None,
    explicit_selection: bool = False,
) -> RunModelConfig:
    """Validate and freeze a run's model configuration.

    Defaults (no request values) reproduce the historical behavior:
    the configured model, ``settings.llm_thinking_level`` when set,
    otherwise ``medium``.

    ``explicit_selection=True`` means the value came from a user
    request — it must be registry-selectable (free-tier eligible, not
    retired). Values that merely mirror the server configuration
    (admin-controlled ``.env``) are trusted as-is, so a custom
    configured model never stops being usable, and a request can never
    select a paid or unknown-eligibility model.
    """
    resolved_model = (
        model_id.strip()
        if isinstance(model_id, str) and model_id.strip()
        else default_model_id
    )
    if isinstance(thinking_level, str) and thinking_level.strip():
        resolved_thinking = thinking_level.strip().lower()
    else:
        resolved_thinking = (default_thinking_level or "medium").lower()

    if resolved_thinking not in THINKING_LEVELS:
        raise ModelSelectionError(
            "unsupported_thinking_level",
            f"thinking_level must be one of {', '.join(THINKING_LEVELS)}",
        )

    spec = spec_for(resolved_model)
    if spec is None:
        # Not in the registry: an admin-configured custom model.
        # User selections may only name registry IDs — or the
        # configured default itself (the UI submits that as
        # "configured default"); configuration stays trusted and
        # thinking is family-checked.
        if explicit_selection and resolved_model != default_model_id:
            raise ModelSelectionError(
                "unknown_model",
                f"'{resolved_model}' is not a known selectable model",
            )
        if resolved_thinking != "auto" and resolved_thinking not in (
            "low",
            "medium",
            "high",
            "minimal",
        ):  # pragma: no cover — guarded by THINKING_LEVELS above
            raise ModelSelectionError(
                "unsupported_thinking_level",
                f"thinking_level must be one of {', '.join(THINKING_LEVELS)}",
            )
        return RunModelConfig(resolved_model, resolved_thinking)

    if explicit_selection and not is_selectable(spec):
        if spec.free_tier == "paid_only":
            reason = "is not available on the free tier"
        elif spec.free_tier == "unknown":
            reason = "has no confirmed free-tier eligibility"
        else:
            reason = f"has status '{spec.status}'"
        raise ModelSelectionError(
            "model_not_selectable",
            f"{spec.display_name} {reason} and cannot be selected",
        )
    _validate_thinking(spec, resolved_thinking)
    return RunModelConfig(resolved_model, resolved_thinking)


def assert_runnable(model_id: str) -> ModelSpec | None:
    """Fail fast when a persisted selection is retired.

    Returns the (possibly custom, unregistered) spec — ``None`` for
    an admin-custom model — or raises :class:`ModelUnavailableError`
    for a tombstoned registry entry. Never substitutes a model.
    """
    spec = spec_for(model_id)
    if spec is not None and spec.status in TOMBSTONE_STATUSES:
        detail = (
            f" (retired {spec.retired_on})" if spec.retired_on else ""
        )
        raise ModelUnavailableError(
            f"model unavailable: {model_id}{detail} is no longer "
            "usable; this run cannot continue with a different model"
        )
    return spec


def thinking_kwargs(
    model_id: str, *, level: str, verification: bool = False
) -> dict[str, object]:
    """Wire-level thinking configuration for one model + level.

    Returns kwargs for ``ChatGoogleGenerativeAI``:

    - level-controlled family (Gemini 3+): ``{"thinking_level": ...}``
      — only levels that model documents;
    - budget-controlled family (Gemini 2.5): ``{"thinking_budget": N}``
      from the official mapping — ``minimal`` and ``disabled`` are
      different things (disabled is never expressible here);
    - ``auto``: ``{}`` — no thinking parameter, model default;
    - no thinking controls (or unknown family): ``{}``.

    ``verification=True`` keeps the connectivity smoke test cheap:
    the lowest level for level models and budget 0 (thinking off) for
    budget models — a documented constraint of the verification path
    only, never a research-run mapping.

    Never returns both ``thinking_level`` and ``thinking_budget``.
    """
    if verification:
        spec = spec_for(model_id)
        control = spec.thinking_control if spec else control_by_family(model_id)
        if control == "level":
            return {"thinking_level": "low"}
        if control == "budget":
            return {"thinking_budget": 0}
        return {}

    if level == "auto":
        return {}

    spec = spec_for(model_id)
    if spec is not None:
        if spec.thinking_control == "level":
            if level in spec.thinking_levels:
                return {"thinking_level": level}
            # An admin-set level the model does not support must not
            # reach the API as a 400: fall back to the model default.
            return {}
        if spec.thinking_control == "budget":
            budget = BUDGET_BY_LEVEL.get(level)
            return {"thinking_budget": budget} if budget else {}
        return {}

    # Unknown (admin-custom) model: family heuristic only.
    control = control_by_family(model_id)
    if control == "level":
        return {"thinking_level": level}
    if control == "budget":
        budget = BUDGET_BY_LEVEL.get(level)
        return {"thinking_budget": budget} if budget else {}
    return {}


def control_by_family(model_id: str) -> ThinkingControl:
    family = family_of(model_id)
    if family in ("gemini-3", "gemini-4"):
        return "level"
    if family == "gemini-2.5":
        # Budget-controlled. Gemini 2.0 and older have NO thinking
        # controls at all — never send them a budget.
        return "budget"
    return "none"
