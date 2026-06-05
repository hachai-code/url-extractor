// TypeScript types mirroring url-extractor/schemas.py.
// Hand-written; if schemas.py changes, update here in lockstep.

export type EntityType =
  | "person"
  | "organization"
  | "location"
  | "product"
  | "event"
  | "work_of_art"
  | "other";

export type SentimentLabel =
  | "very_negative"
  | "negative"
  | "neutral"
  | "positive"
  | "very_positive";

export type Priority = "low" | "medium" | "high";

export interface Entity {
  name: string;
  type: EntityType;
  mentions: number;
  context: string | null;
  wikipedia_title: string | null;
  confidence: number;
}

export interface Sentiment {
  label: SentimentLabel;
  score: number;
  rationale: string;
  confidence: number;
}

export interface ActionItem {
  description: string;
  priority: Priority;
  deadline: string | null;
  confidence: number;
}

export interface Claim {
  statement: string;
  supporting_quote: string | null;
  is_opinion: boolean;
  confidence: number;
}

export interface PageAnalysis {
  url: string;
  title: string;
  summary: string;
  language: string;
  entities: Entity[];
  sentiment: Sentiment;
  action_items: ActionItem[];
  key_claims: Claim[];
}

export interface ExtractError {
  detail: {
    stage: "fetch" | "extract";
    kind: string | null;
    detail: string;
  };
}
