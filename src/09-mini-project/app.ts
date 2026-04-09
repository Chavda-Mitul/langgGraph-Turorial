/**
 * 09-mini-project/app.ts
 * ──────────────────────
 * Multi-Agent Content Pipeline — Main Application
 *
 * Combines ALL modern LangGraph concepts:
 * - StateGraph with Annotations + MessagesAnnotation.spec
 * - Command routing (goto) — no conditional edges
 * - addNode with { ends } declarations
 * - Multiple agent nodes (Supervisor, Researcher, Writer, Reviewer, Finalizer)
 * - Loops (revision cycle via Command routing)
 * - Tool integration (web search, fact checker)
 * - MemorySaver (short-term, per-thread) + InMemoryStore (long-term, cross-thread)
 * - Streaming (watch pipeline execute in real-time)
 *
 * Run: npx ts-node --esm src/09-mini-project/app.ts
 */

import "dotenv/config";
import { buildContentPipeline } from "./graph.js";

async function main() {
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║       Multi-Agent Content Pipeline (v1.x)               ║");
  console.log("║   Command Routing + InMemoryStore + Agent Handoffs      ║");
  console.log("║   Built with LangGraph v1.2                             ║");
  console.log("╚══════════════════════════════════════════════════════════╝");

  const pipeline = buildContentPipeline();
  const config = { configurable: { thread_id: "content-session-1" } };

  // ── Phase 1: Run the pipeline with streaming ─────────────────────
  console.log("\n📌 RUNNING CONTENT PIPELINE\n");
  console.log("═".repeat(60));

  const stream = await pipeline.stream(
    {
      request: "Write an article about how multi-agent AI systems are revolutionizing software development",
    },
    { ...config, streamMode: "updates" }
  );

  for await (const _update of stream) {
    // Streaming is handled by console.logs in agent functions
  }

  // ── Phase 2: Inspect final state ─────────────────────────────────
  const finalState = await pipeline.getState(config);
  const state = finalState.values;

  console.log("\n" + "═".repeat(60));
  console.log("📊 PIPELINE RESULTS");
  console.log("═".repeat(60));

  console.log("\n📋 Status:", state.status);
  console.log("📝 Iterations:", state.iteration);

  console.log("\n📜 Activity Log:");
  state.log.forEach((entry: string, i: number) => {
    console.log(`   ${i + 1}. ${entry}`);
  });

  console.log("\n" + "─".repeat(60));
  console.log("📄 FINAL CONTENT:");
  console.log("─".repeat(60));
  console.log(state.finalContent || state.draft);

  // ── Phase 3: Second request (demonstrates cross-thread store) ────
  console.log("\n\n" + "═".repeat(60));
  console.log("📌 SECOND REQUEST (new thread, shared long-term memory)");
  console.log("═".repeat(60));

  const config2 = { configurable: { thread_id: "content-session-2" } };

  const result2 = await pipeline.invoke(
    {
      request: "Write a short piece about the benefits of graph-based AI orchestration",
    },
    config2
  );

  console.log("\n📄 SECOND ARTICLE:");
  console.log("─".repeat(60));
  console.log(result2.finalContent || result2.draft);

  // ── Stats ────────────────────────────────────────────────────────
  console.log("\n\n📊 Final Stats:");
  console.log(`   Session 1 — Status: ${state.status}, Iterations: ${state.iteration}`);
  console.log(`   Session 2 — Status: ${result2.status}, Iterations: ${result2.iteration}`);
  console.log("\n✅ Multi-Agent Content Pipeline complete!");
}

main().catch(console.error);
