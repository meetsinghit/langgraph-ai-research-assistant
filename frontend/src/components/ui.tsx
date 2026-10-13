/** Small presentational primitives shared across the app. */
import { useEffect, type ButtonHTMLAttributes, type ReactNode } from "react";

type ButtonVariant = "primary" | "ghost" | "danger" | "subtle";

export function Button({
  variant = "subtle",
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  const base =
    "inline-flex items-center justify-center gap-1.5 rounded-lg text-sm font-medium " +
    "transition-colors disabled:opacity-50 disabled:cursor-not-allowed px-3 py-1.5";
  const variants: Record<ButtonVariant, string> = {
    primary: "bg-accent text-white hover:opacity-90",
    ghost: "text-ink-muted hover:text-ink hover:bg-surface-2",
    subtle: "bg-surface-2 text-ink hover:bg-border/60 border border-border",
    danger: "bg-danger text-white hover:opacity-90",
  };
  return (
    <button
      type="button"
      className={`${base} ${variants[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span
      className="spinner inline-block h-3.5 w-3.5 rounded-full border-2 border-border border-t-accent align-middle"
      role="status"
      aria-label={label ?? "Loading"}
    />
  );
}

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "success" | "warning" | "danger" | "accent";
  children: ReactNode;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-surface-2 text-ink-muted border-border",
    success: "border-success/40 text-success bg-success/10",
    warning: "border-warning/40 text-warning bg-warning/10",
    danger: "border-danger/40 text-danger bg-danger/10",
    accent: "border-accent/40 text-accent bg-accent-soft",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Modal({
  open,
  title,
  children,
  onClose,
  labelledBy = "modal-title",
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  onClose: () => void;
  labelledBy?: string;
}) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        className="w-full max-w-md rounded-xl border border-border bg-surface p-5 shadow-xl"
      >
        <h2 id={labelledBy} className="mb-3 text-base font-semibold">
          {title}
        </h2>
        {children}
      </div>
    </div>
  );
}

export function Drawer({
  open,
  side,
  label,
  children,
  onClose,
}: {
  open: boolean;
  side: "left" | "right";
  label: string;
  children: ReactNode;
  onClose: () => void;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 lg:hidden">
      <div
        className="absolute inset-0 bg-black/40"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-label={label}
        className={`absolute inset-y-0 ${side === "left" ? "left-0" : "right-0"} w-[85%] max-w-xs overflow-y-auto border-border bg-surface shadow-xl ${
          side === "left" ? "border-r" : "border-l"
        }`}
      >
        {children}
      </div>
    </div>
  );
}

export function Tabs({
  tabs,
  active,
  onChange,
  ariaLabel,
}: {
  tabs: Array<{ id: string; label: string }>;
  active: string;
  onChange: (id: string) => void;
  ariaLabel: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className="flex border-b border-border"
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          type="button"
          aria-selected={active === tab.id}
          onClick={() => onChange(tab.id)}
          className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
            active === tab.id
              ? "border-b-2 border-accent text-ink"
              : "border-b-2 border-transparent text-ink-muted hover:text-ink"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

// ---- icons -----------------------------------------------------------------

export const Icons = {
  plus: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M8 3v10M3 8h10" strokeLinecap="round" />
    </svg>
  ),
  trash: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 8.2a1 1 0 001 .8h3.6a1 1 0 001-.8l.7-8.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  close: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
    </svg>
  ),
  check: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M3.5 8.5l3 3 6-7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  copy: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="5.5" y="5.5" width="7" height="8" rx="1.2" />
      <path d="M10.5 5.5v-2A1.5 1.5 0 009 2H4.5A1.5 1.5 0 003 3.5V10a1.5 1.5 0 001.5 1.5h1" />
    </svg>
  ),
  stop: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor">
      <rect x="4" y="4" width="8" height="8" rx="1.5" />
    </svg>
  ),
  menu: (
    <svg viewBox="0 0 16 16" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M2.5 4h11M2.5 8h11M2.5 12h11" strokeLinecap="round" />
    </svg>
  ),
  panel: (
    <svg viewBox="0 0 16 16" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="2" y="3" width="12" height="10" rx="1.5" />
      <path d="M10 3v10" />
    </svg>
  ),
  refresh: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M13 8a5 5 0 11-1.5-3.6M13 2.5V5h-2.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  sun: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M12.6 3.4l-1.1 1.1M4.5 11.5l-1.1 1.1" strokeLinecap="round" />
    </svg>
  ),
  moon: (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M13 9.5A5.5 5.5 0 016.5 3a5.5 5.5 0 106.5 6.5z" strokeLinejoin="round" />
    </svg>
  ),
};
