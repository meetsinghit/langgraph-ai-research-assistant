/** Chat-style question input: Enter submits, Shift+Enter newlines.
 *  Optional `controls` render above the textarea inside the same
 *  form (research settings, 4E.1/4E.3). */
import { useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { Button } from "./ui";

const MAX_QUESTION = 2000;

export function QuestionInput({
  onSubmit,
  busy,
  disabled,
  controls,
}: {
  onSubmit: (question: string) => void;
  busy: boolean;
  disabled?: boolean;
  controls?: ReactNode;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement | null>(null);
  const canSubmit = value.trim().length > 0 && !busy && !disabled;

  const submit = () => {
    if (!canSubmit) return;
    onSubmit(value);
    setValue("");
    if (ref.current) ref.current.style.height = "auto";
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form
      className="rounded-xl border border-border bg-surface p-2 shadow-sm focus-within:border-accent"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      {controls}
      <div className="flex items-end gap-2">
        <label htmlFor="question-input" className="sr-only">
          Research question
        </label>
        <textarea
          id="question-input"
          ref={ref}
          rows={1}
          value={value}
          maxLength={MAX_QUESTION}
          placeholder="Ask a research question… (Enter to submit, Shift+Enter for a new line)"
          onChange={(event) => {
            setValue(event.target.value);
            event.target.style.height = "auto";
            event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`;
          }}
          onKeyDown={onKeyDown}
          className="max-h-[200px] min-h-[38px] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-ink outline-none placeholder:text-ink-muted"
          aria-describedby="question-hint"
        />
        <Button
          variant="primary"
          type="submit"
          disabled={!canSubmit}
          aria-label="Start research"
        >
          Research
        </Button>
        <span id="question-hint" className="sr-only">
          Press Enter to submit the question. Press Shift and Enter for a new
          line.
        </span>
        {value.length > MAX_QUESTION - 200 && (
          <span className="pb-1 text-[11px] text-ink-muted" aria-live="polite">
            {value.length}/{MAX_QUESTION}
          </span>
        )}
      </div>
    </form>
  );
}
