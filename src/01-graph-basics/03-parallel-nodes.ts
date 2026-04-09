/**
 * 01-graph-basics/03-parallel-nodes.ts
 * ─────────────────────────────────────
 * Parallel Nodes: Fan-out to multiple nodes, then fan-in.
 *
 * Key Concepts:
 * - Multiple edges from one node = parallel execution
 * - Fan-out: One node feeds multiple nodes
 * - Fan-in: Multiple nodes feed into one node
 * - Reducers: How parallel results merge into state
 *
 * Flow:
 *                 ┌──→ pros_analyst ──┐
 *   START ──→ researcher               ──→ summarizer ──→ END
 *                 └──→ cons_analyst ──┘
 *
 * Run: npx ts-node --esm src/01-graph-basics/03-parallel-nodes.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  Annotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 300,
});

// ── 1. State with fields for parallel results ──────────────────────
const DebateState = Annotation.Root({
  topic: Annotation<string>(),
  research: Annotation<string>(),
  pros: Annotation<string>(),
  cons: Annotation<string>(),
  summary: Annotation<string>(),
});

// ── 2. Nodes ───────────────────────────────────────────────────────
async function researcher(state: typeof DebateState.State) {
  console.log("📚 Researching:", state.topic);
  const res = await model.invoke(`Give a brief overview of: ${state.topic}`);
  return { research: res.content as string };
}

async function prosAnalyst(state: typeof DebateState.State) {
  console.log("👍 Analyzing pros...");
  const res = await model.invoke(
    `Based on this research, list 3 PROS/advantages:\n\n${state.research}`
  );
  return { pros: res.content as string };
}

async function consAnalyst(state: typeof DebateState.State) {
  console.log("👎 Analyzing cons...");
  const res = await model.invoke(
    `Based on this research, list 3 CONS/disadvantages:\n\n${state.research}`
  );
  return { cons: res.content as string };
}

async function summarizer(state: typeof DebateState.State) {
  console.log("📝 Summarizing...");
  const res = await model.invoke(
    `Create a balanced summary from:\n\nPROS:\n${state.pros}\n\nCONS:\n${state.cons}`
  );
  return { summary: res.content as string };
}

// ── 3. Build graph with fan-out and fan-in ─────────────────────────
const graph = new StateGraph(DebateState)
  .addNode("researcher", researcher)
  .addNode("pros_analyst", prosAnalyst)
  .addNode("cons_analyst", consAnalyst)
  .addNode("summarizer", summarizer)
  // Sequential: START → researcher
  .addEdge(START, "researcher")
  // Fan-out: researcher → both analysts (parallel!)
  .addEdge("researcher", "pros_analyst")
  .addEdge("researcher", "cons_analyst")
  // Fan-in: both analysts → summarizer
  .addEdge("pros_analyst", "summarizer")
  .addEdge("cons_analyst", "summarizer")
  // End
  .addEdge("summarizer", END)
  .compile();

// ── 4. Run ─────────────────────────────────────────────────────────
const result = await graph.invoke({
  topic: "Using AI agents in production applications",
});

console.log("\n=== Parallel Nodes Result ===");
console.log("\n👍 Pros:\n", result.pros);
console.log("\n👎 Cons:\n", result.cons);
console.log("\n📝 Summary:\n", result.summary);
