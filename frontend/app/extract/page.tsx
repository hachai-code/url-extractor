"use client";

import { useState } from "react";
import type {
  ActionItem,
  Claim,
  Entity,
  ExtractError,
  PageAnalysis,
} from "./types";

const TABS = ["Summary", "Entities", "Claims", "Action Items"] as const;
type Tab = (typeof TABS)[number];

const API_URL = "http://localhost:8000/extract";

export default function ExtractPage() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PageAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("Summary");

  async function analyze(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const body = await response.json();
      if (!response.ok) {
        const err = body as ExtractError;
        setError(
          `${err.detail.stage} failed${err.detail.kind ? ` (${err.detail.kind})` : ""}: ${err.detail.detail}`,
        );
      } else {
        setResult(body as PageAnalysis);
        setActiveTab("Summary");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "unknown error");
    } finally {
      setLoading(false);
    }
  }

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
          disabled={loading}
          className="rounded bg-black px-4 py-2 text-white disabled:opacity-50"
        >
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </form>

      {error && (
        <div className="rounded border border-red-200 bg-red-50 p-4 text-red-800">
          {error}
        </div>
      )}

      {result && (
        <section>
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
          <div className="py-4">
            {activeTab === "Summary" && <SummaryView a={result} />}
            {activeTab === "Entities" && (
              <EntitiesView entities={result.entities} />
            )}
            {activeTab === "Claims" && <ClaimsView claims={result.key_claims} />}
            {activeTab === "Action Items" && (
              <ActionsView items={result.action_items} />
            )}
          </div>
        </section>
      )}
    </main>
  );
}

function SummaryView({ a }: { a: PageAnalysis }) {
  return (
    <div className="space-y-3">
      <div>
        <h2 className="font-medium">{a.title}</h2>
        <p className="break-all text-sm text-gray-500">{a.url}</p>
      </div>
      <p className="leading-relaxed">{a.summary}</p>
      <div className="border-t border-gray-200 pt-3 text-sm">
        <div>
          Sentiment: <strong>{a.sentiment.label}</strong> (score{" "}
          {a.sentiment.score.toFixed(2)}, confidence{" "}
          {a.sentiment.confidence.toFixed(2)})
        </div>
        <div className="mt-1 italic text-gray-700">
          {a.sentiment.rationale}
        </div>
      </div>
    </div>
  );
}

function EntitiesView({ entities }: { entities: Entity[] }) {
  if (entities.length === 0) {
    return <p className="text-gray-500">No entities extracted.</p>;
  }
  return (
    <ul className="space-y-3">
      {entities.map((e, i) => (
        <li key={i} className="rounded border border-gray-200 p-3">
          <div className="flex items-baseline justify-between gap-2">
            <strong>{e.name}</strong>
            <span className="shrink-0 text-xs text-gray-500">
              {e.type} · conf {e.confidence.toFixed(2)} · {e.mentions}×
            </span>
          </div>
          {e.context && (
            <p className="mt-1 text-sm text-gray-700">{e.context}</p>
          )}
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

function ClaimsView({ claims }: { claims: Claim[] }) {
  if (claims.length === 0) {
    return <p className="text-gray-500">No key claims.</p>;
  }
  return (
    <ul className="space-y-3">
      {claims.map((c, i) => (
        <li key={i} className="rounded border border-gray-200 p-3">
          <div className="flex items-baseline justify-between gap-2">
            <p>{c.statement}</p>
            <span className="shrink-0 text-xs text-gray-500">
              {c.is_opinion ? "opinion" : "fact"} · conf{" "}
              {c.confidence.toFixed(2)}
            </span>
          </div>
          {c.supporting_quote && (
            <blockquote className="mt-2 border-l-2 border-gray-300 pl-3 text-sm italic text-gray-600">
              “{c.supporting_quote}”
            </blockquote>
          )}
        </li>
      ))}
    </ul>
  );
}

function ActionsView({ items }: { items: ActionItem[] }) {
  if (items.length === 0) {
    return (
      <p className="text-gray-500">
        No action items — this page doesn’t ask the reader to do anything.
      </p>
    );
  }
  const priorityColor = {
    high: "bg-red-100 text-red-800",
    medium: "bg-yellow-100 text-yellow-800",
    low: "bg-gray-100 text-gray-700",
  };
  return (
    <ul className="space-y-3">
      {items.map((a, i) => (
        <li key={i} className="rounded border border-gray-200 p-3">
          <div className="flex items-baseline justify-between gap-2">
            <p>{a.description}</p>
            <span
              className={`shrink-0 rounded px-2 py-0.5 text-xs ${priorityColor[a.priority]}`}
            >
              {a.priority}
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
