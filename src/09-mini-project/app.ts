/**
 * 09-mini-project/app.ts
 * ──────────────────────
 * Multi-Agent Content Pipeline — Main Application
 *
 * Combines ALL LangGraph concepts:
 * - StateGraph with custom Annotations and reducers
 * - Multiple agent nodes (Supervisor, Researcher, Writer, Reviewer, Finalizer)
 * - Conditional edges (supervisor routing)
 * - Loops (revision cycle: writer → reviewer → writer)
 * - Tool integration (web search, fact checker)
 * - Memory/checkpoints (MemorySaver with thread_id)
 * - Streaming (watch the pipeline execute in real-time)
 *
 * Run: npx ts-node --esm src/09-mini-project/app.ts
 */

import "dotenv/config";
import { buildContentPipeline } from "./graph.js";

async function main() {
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║       Multi-Agent Content Pipeline                      ║");
  console.log("║   Supervisor + Researcher + Writer + Reviewer           ║");
  console.log("║   Built with LangGraph                                  ║");
  console.log("╚═══��═══════════════════════════════════════════════════��══╝");

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

  for await (const update of stream) {
    const [nodeName] = Object.entries(update);
    // The streaming is handled by console.logs in agent functions
  }

  // ── Phase 2: Inspect final state ─────────────────────────────────
  const finalState = await pipeline.getState(config);
  const state = finalState.values;

  console.log("\n" + "═".repeat(60));
  console.log("�� PIPELINE RESULTS");
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

  // ── Phase 3: Run a second request (same session) ─────────────────
  console.log("\n\n" + "═".repeat(60));
  console.log("📌 SECOND REQUEST (same session)");
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
  console.log(`   Session 1 - Status: ${state.status}, Iterations: ${state.iteration}`);
  console.log(`   Session 2 - Status: ${result2.status}, Iterations: ${result2.iteration}`);
  console.log("\n✅ Multi-Agent Content Pipeline complete!");
}

main().catch(console.error);
