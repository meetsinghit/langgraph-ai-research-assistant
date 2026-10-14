/**
 * Composer research settings (4E.1/4E.3): model, thinking effort,
 * research depth and evidence policy — all frozen into the run at
 * creation, so historical answers keep their original settings.
 *
 * Capability-aware (WS-B): the thinking selector only offers levels
 * the selected model actually supports (per the backend catalogue),
 * and inherited selections that became unsupported are corrected to
 * the model's default — the UI never offers a combination the
 * backend must reject.
 *
 * A11y: every select has a visible <label>; state is conveyed by the
 * option text, not color. Disabled while a run is in flight.
 */
import { useEffect } from "react";
import { useStore } from "../state/store";
import type { EvidencePolicy, ResearchDepth, ThinkingLevel } from "../api/types";

const DEPTH_OPTIONS: { value: ResearchDepth; label: string }[] = [
  { value: "quick", label: "Quick — fewer sources" },
  { value: "standard", label: "Standard" },
  { value: "deep", label: "Deep — more sources" },
];

const EVIDENCE_OPTIONS: { value: EvidencePolicy; label: string }[] = [
  { value: "balanced", label: "Balanced — snippets allowed" },
  { value: "verified", label: "Verified — fetched pages only" },
];

const THINKING_LABELS: Record<ThinkingLevel, string> = {
  auto: "Auto",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
};

const LEVEL_ORDER: ThinkingLevel[] = ["auto", "minimal", "low", "medium", "high"];

function Field({
  id,
  label,
  children,
}: {
  id: string;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 sm:flex-none">
      <label htmlFor={id} className="text-[10px] font-medium text-ink-muted">
        {label}
      </label>
      {children}
    </div>
  );
}

const SELECT_CLASS =
  "h-8 max-w-full min-w-0 rounded-lg border border-border bg-surface px-2 text-xs text-ink outline-none focus-visible:border-accent disabled:opacity-60 sm:w-auto";

export function ResearchControls({ disabled }: { disabled?: boolean }) {
  const store = useStore();
  const {
    models,
    modelsStatus,
    defaultModelId,
    defaultThinkingLevel,
    preferences,
    setPreference,
  } = store;

  const catalogReady = modelsStatus === "ready";
  const selected = preferences.modelId
    ? (models.find((m) => m.model_id === preferences.modelId) ?? null)
    : null;
  // "Default model" inherits the default model's capabilities.
  const activeSpec = selected ?? models.find((m) => m.model_id === defaultModelId);
  const thinkingInfo = activeSpec?.thinking ?? null;
  const thinkingSupported =
    catalogReady &&
    thinkingInfo != null &&
    thinkingInfo.control !== "none" &&
    (preferences.modelId === null || selected != null);
  const levels = thinkingSupported
    ? LEVEL_ORDER.filter((level) => thinkingInfo?.levels.includes(level))
    : [];

  // Sanitize inherited selections once the catalogue is known: a
  // model that is no longer selectable falls back to the configured
  // default; a thinking level the model does not support falls back
  // to that model's default.
  useEffect(() => {
    if (!catalogReady) return;
    if (
      preferences.modelId !== null &&
      !models.some((m) => m.model_id === preferences.modelId && m.selectable)
    ) {
      setPreference({ modelId: null });
      return;
    }
    if (
      preferences.thinkingLevel != null &&
      (!thinkingSupported || !levels.includes(preferences.thinkingLevel))
    ) {
      setPreference({ thinkingLevel: null });
    }
  }, [
    catalogReady,
    models,
    preferences.modelId,
    preferences.thinkingLevel,
    thinkingSupported,
    levels,
    setPreference,
  ]);

  const selectable = models.filter(
    (m) => m.selectable && m.model_id !== defaultModelId,
  );
  const defaultName =
    models.find((m) => m.model_id === defaultModelId)?.display_name ??
    defaultModelId ??
    "configured";
  const defaultLevelLabel =
    thinkingInfo?.default ?? defaultThinkingLevel ?? null;

  return (
    <div
      className="mb-2 flex flex-wrap items-end gap-2"
      role="group"
      aria-label="Research settings"
    >
      <Field id="control-model" label="Model">
        <select
          id="control-model"
          className={SELECT_CLASS}
          value={preferences.modelId ?? ""}
          disabled={disabled || modelsStatus === "loading"}
          onChange={(event) =>
            setPreference({ modelId: event.target.value || null })
          }
        >
          <option value="">
            {catalogReady ? `Default (${defaultName})` : "Default model"}
          </option>
          {selectable.map((model) => (
            <option key={model.model_id} value={model.model_id}>
              {model.display_name}
            </option>
          ))}
        </select>
      </Field>

      <Field id="control-thinking" label="Thinking">
        <select
          id="control-thinking"
          className={SELECT_CLASS}
          value={preferences.thinkingLevel ?? ""}
          disabled={disabled || !thinkingSupported}
          onChange={(event) =>
            setPreference({
              thinkingLevel: (event.target.value || null) as ThinkingLevel | null,
            })
          }
        >
          <option value="">
            {defaultLevelLabel ? `Default (${defaultLevelLabel})` : "Default"}
          </option>
          {levels.map((level) => (
            <option key={level} value={level}>
              {THINKING_LABELS[level]}
            </option>
          ))}
        </select>
      </Field>

      <Field id="control-depth" label="Depth">
        <select
          id="control-depth"
          className={SELECT_CLASS}
          value={preferences.depth}
          disabled={disabled}
          onChange={(event) =>
            setPreference({ depth: event.target.value as ResearchDepth })
          }
        >
          {DEPTH_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </Field>

      <Field id="control-evidence" label="Evidence">
        <select
          id="control-evidence"
          className={SELECT_CLASS}
          value={preferences.evidence}
          disabled={disabled}
          onChange={(event) =>
            setPreference({
              evidence: event.target.value as EvidencePolicy,
            })
          }
        >
          {EVIDENCE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </Field>

      {modelsStatus === "failed" && (
        <p className="pb-1 text-[10px] text-ink-muted">
          Model list unavailable — the configured default will be used.
        </p>
      )}
      {catalogReady && !thinkingSupported && preferences.modelId !== null && (
        <p className="pb-1 text-[10px] text-ink-muted">
          This model does not expose thinking controls.
        </p>
      )}
    </div>
  );
}
