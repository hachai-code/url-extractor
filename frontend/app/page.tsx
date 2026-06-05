"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ActionItem,
  Claim,
  Entity,
  ExtractError,
  PageAnalysis,
} from "./types";

const TABS = ["Summary", "Entities", "Claims", "Action Items"] as const;
type Tab = (typeof TABS)[number];
type Phase = "idle" | "streaming" | "done" | "error";

const API_BASE = "http://localhost:8000";

export default function HomePage() {
  const [url, setUrl] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [partial, setPartial] = useState<Partial<PageAnalysis> | null>(null);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("Summary");
  const [showRaw, setShowRaw] = useState(false);
  const [copied, setCopied] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Load a shared analysis from ?id=... on first render.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    if (!id) return;
    setPhase("streaming");
    fetch(`${API_BASE}/analyses/${id}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r)))
      .then((data: { url: string; analysis: PageAnalysis }) => {
        setUrl(data.url);
        setPartial(data.analysis);
        setSavedId(id);
        setPhase("done");
      })
      .catch(() => {
        setError("Could not load shared analysis (link may be expired).");
        setPhase("error");
      });
  }, []);

  const analyze = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      setPhase("streaming");
      setPartial(null);
      setSavedId(null);
      setError(null);
      setActiveTab("Summary");

      try {
        const response = await fetch(`${API_BASE}/extract/stream`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ url }),
          signal: ac.signal,
        });

        if (!response.ok) {
          const body = (await response.json()) as ExtractError;
          setError(
            `${body.detail.stage} failed${body.detail.kind ? ` (${body.detail.kind})` : ""}: ${body.detail.detail}`,
          );
          setPhase("error");
          return;
        }

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let sep;
          while ((sep = buffer.indexOf("\n\n")) >= 0) {
            const block = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);

            let eventName = "message";
            let data = "";
            for (const line of block.split("\n")) {
              if (line.startsWith("event: ")) eventName = line.slice(7).trim();
              else if (line.startsWith("data: ")) data += line.slice(6);
            }

            if (eventName === "message" && data) {
              setPartial(JSON.parse(data) as Partial<PageAnalysis>);
            } else if (eventName === "done") {
              const meta = JSON.parse(data || "{}") as { id?: string | null };
              if (meta.id) {
                setSavedId(meta.id);
                window.history.replaceState(null, "", `?id=${meta.id}`);
              }
              setPhase("done");
            } else if (eventName === "error") {
              const err = JSON.parse(data) as { kind?: string; detail?: string };
              setError(err.detail || "extraction failed");
              setPhase("error");
            }
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError(err instanceof Error ? err.message : "unknown error");
        setPhase("error");
      }
    },
    [url],
  );

  const onShare = useCallback(async () => {
    if (!savedId) return;
    const link = `${window.location.origin}/?id=${savedId}`;
    await navigator.clipboard.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [savedId]);

  const streaming = phase === "streaming";

  return (
    <main className="mx-auto max-w-3xl space-y-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold">URL extractor</h1>
        <p className="text-sm text-gray-600">
          Paste any URL to get a structured analysis: summary, entities, claims,
          action items.
        </p>
      </header>

      <form onSubmit={analyze} className="flex gap-2">
        <input
          type="url"
          required
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://..."
          className="flex-1 rounded border border-gray-300 px-3 py-2"
        />
        <button
          type="submit"
          disabled={streaming}
          className="rounded bg-black px-4 py-2 text-white disabled:opacity-50"
        >
          {streaming ? "Analyzing…" : "Analyze"}
        </button>
      </form>

      {error && (
        <div className="rounded border border-red-200 bg-red-50 p-4 text-red-800">
          {error}
        </div>
      )}

      {(streaming || partial) && (
        <section className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <nav className="flex border-b border-gray-200">
              {TABS.map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`-mb-px border-b-2 px-4 py-2 text-sm transition-colors ${
                    activeTab === t
                      ? "border-black font-medium"
                      : "border-transparent text-gray-500 hover:text-gray-700"
                  }`}
                >
                  {t}
                </button>
              ))}
            </nav>
            <div className="flex shrink-0 gap-2">
              <button
                onClick={() => setShowRaw((v) => !v)}
                className="rounded border border-gray-200 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
              >
                {showRaw ? "Hide raw JSON" : "View raw JSON"}
              </button>
              {savedId && (
                <button
                  onClick={onShare}
                  className="rounded border border-gray-200 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
                >
                  {copied ? "Copied!" : "Share this analysis"}
                </button>
              )}
            </div>
          </div>

          {showRaw ? (
            <pre className="overflow-x-auto rounded bg-gray-50 p-4 text-xs">
              {JSON.stringify(partial, null, 2)}
            </pre>
          ) : (
            <div className="py-2">
              {activeTab === "Summary" && (
                <SummaryView a={partial} streaming={streaming} />
              )}
              {activeTab === "Entities" && (
                <EntitiesView
                  entities={(partial?.entities ?? []) as Entity[]}
                  streaming={streaming}
                />
              )}
              {activeTab === "Claims" && (
                <ClaimsView
                  claims={(partial?.key_claims ?? []) as Claim[]}
                  streaming={streaming}
                />
              )}
              {activeTab === "Action Items" && (
                <ActionsView
                  items={(partial?.action_items ?? []) as ActionItem[]}
                  streaming={streaming}
                />
              )}
            </div>
          )}
        </section>
      )}
    </main>
  );
}

function SkeletonLine({ width = "w-full" }: { width?: string }) {
  return <div className={`h-3 animate-pulse rounded bg-gray-200 ${width}`} />;
}

function SkeletonCard() {
  return (
    <li className="space-y-2 rounded border border-gray-200 p-3">
      <SkeletonLine width="w-1/3" />
      <SkeletonLine width="w-5/6" />
      <SkeletonLine width="w-2/3" />
    </li>
  );
}

function SummaryView({
  a,
  streaming,
}: {
  a: Partial<PageAnalysis> | null;
  streaming: boolean;
}) {
  if (!a) return null;
  return (
    <div className="space-y-3">
      <div>
        {a.title ? (
          <h2 className="font-medium">{a.title}</h2>
        ) : (
          <SkeletonLine width="w-1/2" />
        )}
        {a.url && <p className="break-all text-sm text-gray-500">{a.url}</p>}
      </div>
      {a.summary ? (
        <p className="leading-relaxed">{a.summary}</p>
      ) : (
        <div className="space-y-2">
          <SkeletonLine />
          <SkeletonLine />
          <SkeletonLine width="w-3/4" />
        </div>
      )}
      <div className="border-t border-gray-200 pt-3 text-sm">
        {a.sentiment?.label ? (
          <>
            <div>
              Sentiment: <strong>{a.sentiment.label}</strong>
              {typeof a.sentiment.score === "number" && (
                <>
                  {" "}
                  (score {a.sentiment.score.toFixed(2)}
                  {typeof a.sentiment.confidence === "number" &&
                    `, confidence ${a.sentiment.confidence.toFixed(2)}`}
                  )
                </>
              )}
            </div>
            {a.sentiment.rationale && (
              <div className="mt-1 italic text-gray-700">
                {a.sentiment.rationale}
              </div>
            )}
          </>
        ) : streaming ? (
          <SkeletonLine width="w-2/3" />
        ) : null}
      </div>
    </div>
  );
}

function EntitiesView({
  entities,
  streaming,
}: {
  entities: Entity[];
  streaming: boolean;
}) {
  if (entities.length === 0) {
    return streaming ? (
      <ul className="space-y-3">
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </ul>
    ) : (
      <p className="text-gray-500">No entities extracted.</p>
    );
  }
  return (
    <ul className="space-y-3">
      {entities.map((e, i) => (
        <li key={i} className="rounded border border-gray-200 p-3">
          <div className="flex items-baseline justify-between gap-2">
            <strong>{e.name ?? ""}</strong>
            <span className="shrink-0 text-xs text-gray-500">
              {e.type ?? "…"}
              {typeof e.confidence === "number" &&
                ` · conf ${e.confidence.toFixed(2)}`}
              {typeof e.mentions === "number" && ` · ${e.mentions}×`}
            </span>
          </div>
          {e.context && <p className="mt-1 text-sm text-gray-700">{e.context}</p>}
          {e.wikipedia_title && (
            <a
              href={`https://en.wikipedia.org/wiki/${encodeURIComponent(e.wikipedia_title)}`}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-block text-xs text-blue-600 underline"
            >
              Wikipedia: {e.wikipedia_title}
            </a>
          )}
        </li>
      ))}
    </ul>
  );
}

function ClaimsView({
  claims,
  streaming,
}: {
  claims: Claim[];
  streaming: boolean;
}) {
  if (claims.length === 0) {
    return streaming ? (
      <ul className="space-y-3">
        <SkeletonCard />
        <SkeletonCard />
      </ul>
    ) : (
      <p className="text-gray-500">No key claims.</p>
    );
  }
  return (
    <ul className="space-y-3">
      {claims.map((c, i) => (
        <li key={i} className="rounded border border-gray-200 p-3">
          <div className="flex items-baseline justify-between gap-2">
            <p>{c.statement ?? ""}</p>
            <span className="shrink-0 text-xs text-gray-500">
              {c.is_opinion ? "opinion" : "fact"}
              {typeof c.confidence === "number" &&
                ` · conf ${c.confidence.toFixed(2)}`}
            </span>
          </div>
          {c.supporting_quote && (
            <blockquote className="mt-2 border-l-2 border-gray-300 pl-3 text-sm italic text-gray-600">
              &ldquo;{c.supporting_quote}&rdquo;
            </blockquote>
          )}
        </li>
      ))}
    </ul>
  );
}

function ActionsView({
  items,
  streaming,
}: {
  items: ActionItem[];
  streaming: boolean;
}) {
  if (items.length === 0) {
    return streaming ? (
      <ul className="space-y-3">
        <SkeletonCard />
      </ul>
    ) : (
      <p className="text-gray-500">
        No action items — this page doesn&rsquo;t ask the reader to do anything.
      </p>
    );
  }
  const priorityColor: Record<string, string> = {
    high: "bg-red-100 text-red-800",
    medium: "bg-yellow-100 text-yellow-800",
    low: "bg-gray-100 text-gray-700",
  };
  return (
    <ul className="space-y-3">
      {items.map((a, i) => (
        <li key={i} className="rounded border border-gray-200 p-3">
          <div className="flex items-baseline justify-between gap-2">
            <p>{a.description ?? ""}</p>
            <span
              className={`shrink-0 rounded px-2 py-0.5 text-xs ${priorityColor[a.priority ?? "medium"]}`}
            >
              {a.priority ?? "—"}
            </span>
          </div>
          {a.deadline && (
            <p className="mt-1 text-xs text-gray-500">Deadline: {a.deadline}</p>
          )}
        </li>
      ))}
    </ul>
  );
}
