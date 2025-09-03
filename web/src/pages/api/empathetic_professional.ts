import type { NextApiRequest, NextApiResponse } from "next";
import { Agent, fetch as undiciFetch } from "undici";

// Per-request undici Agent: disable header/body timeouts (wait until server responds)
const agent = new Agent({
  connect: { timeout: 60_000 },
  headersTimeout: 0,
  bodyTimeout: 0,
});

const RAW_API_BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:8000";
const API_BASE_URL = RAW_API_BASE_URL.replace("localhost", "127.0.0.1");
const USE_FASTAPI_V2 = (process.env.USE_FASTAPI_V2 || "true").toLowerCase() === "true";

// Overall request timeout (AbortController). 0 = disabled (wait indefinitely).
const DEFAULT_TIMEOUT_MS = Number(process.env.API_TIMEOUT_MS ?? 0);
const MIN_TIMEOUT_MS = 60_000;
const MAX_TIMEOUT_MS = 86_400_000; // 24h

type Payload = {
  question: string;
  type?: string;
  history?: unknown[];
  session_id?: string;
  weekly_goal?: unknown;
  feasibility?: unknown;
  anxiety_level?: unknown;
  user_profile?: unknown;
};

function buildOptionalTimeoutSignal(ms: number) {
  if (!Number.isFinite(ms) || ms <= 0) return { signal: undefined as AbortSignal | undefined, cancel: () => {} };
  const safe = Math.min(Math.max(ms, MIN_TIMEOUT_MS), MAX_TIMEOUT_MS);
  const native = (AbortSignal as any).timeout?.(safe);
  if (native) return { signal: native as AbortSignal, cancel: () => {} };
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), safe);
  return { signal: ctrl.signal, cancel: () => clearTimeout(t) };
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const {
    question,
    type,
    history,
    session_id,
    weekly_goal,
    feasibility,
    anxiety_level,
    user_profile,
  } = (req.body || {}) as Payload;

  if (!question || typeof question !== "string") {
    return res.status(400).json({ error: "Question is required and must be a string" });
    }

  // Respect style selector: the UI may send "empathetic_professional"; map to "balanced"
  const normalizedType = type === "empathetic_professional" ? "balanced" : (type || "balanced");

  const override = req.query.timeoutMs ? Number(req.query.timeoutMs) : undefined;
  const API_TIMEOUT_MS = Number.isFinite(override as number) ? (override as number) : DEFAULT_TIMEOUT_MS;

  const path = USE_FASTAPI_V2 ? "/api/v2/empathetic_professional" : "/api/empathetic_professional";
  const url = `${API_BASE_URL}${path}`;

  console.log("Next.js API (Empathetic Professional): Processing request:", {
    url,
    question: question.substring(0, 50),
    type: normalizedType,
    historyLength: Array.isArray(history) ? history.length : 0,
    timeoutMs: API_TIMEOUT_MS,
    v2: USE_FASTAPI_V2,
  });

  const { signal, cancel } = buildOptionalTimeoutSignal(API_TIMEOUT_MS);

  try {
    const init: any = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({
        question,
        type: normalizedType,
        history: history || [],
        session_id,
        weekly_goal,
        feasibility,
        anxiety_level,
        user_profile,
      }),
      cache: "no-store",
      dispatcher: agent,
    };
    if (signal) init.signal = signal;

    const r = await undiciFetch(url, init);
    console.log("Next.js API: FastAPI response status:", r.status);

    const raw = await r.text();
    console.log("Next.js API: Raw response text length:", raw.length);

    let data: any = null;
    try {
      data = raw ? JSON.parse(raw) : null;
    } catch (e) {
      console.error("Next.js API: JSON parse error:", e);
      return res.status(502).json({ error: "Failed to parse backend response", raw });
    }

    if (!r.ok) {
      const detail = data?.detail || data?.error || data || "Backend error";
      console.error(`Next.js API: Backend error ${r.status}:`, detail);
      return res.status(r.status).json({ error: detail });
    }

    return res.status(200).json({
      answer: data?.answer || "No response received",
      question: data?.question,
      tone: data?.tone,
      status: data?.status,
      meta: data?.meta,
    });
  } catch (err: any) {
    const cause: any = err?.cause || {};
    console.error("Next.js API: Fetch error detail:", {
      name: err?.name,
      message: err?.message,
      code: cause?.code,
      errno: cause?.errno,
      stack: err?.stack,
    });

    if (err?.name === "AbortError" || /aborted|timeout/i.test(err?.message)) {
      return res.status(504).json({ error: "Gateway Timeout", message: err?.message });
    }
    return res.status(502).json({
      error: "Upstream network error",
      message: err?.message,
      code: cause?.code,
    });
  } finally {
    cancel();
  }
}
