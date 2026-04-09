/**
 * 07-streaming/01-stream-modes.ts
 * ─────────────────────────────────
 * Streaming: Watch graph execution in real-time.
 *
 * Key Concepts:
 * - graph.stream(): Returns an async iterator of updates
 * - streamMode "values": Get full state after each node
 * - streamMode "updates": Get only the changes from each node
 * - Useful for: progress indicators, real-time UIs, debugging
 *
 * Run: npx ts-node --esm src/07-streaming/01-stream-modes.ts
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
  maxTokens: 150,
});

// ── 1. State ───────────────────────────────────────────────────────
const PipelineState = Annotation.Root({
  topic: Annotation<string>(),
  research: Annotation<string>(),
  outline: Annotation<string>(),
  article: Annotation<string>(),
});

// ── 2. Nodes ───────────────────────────────────────────────────────
async function researcher(state: typeof PipelineState.State) {
  const res = await model.invoke(`Research briefly: ${state.topic}`);
  return { research: res.content as string };
}

async function outliner(state: typeof PipelineState.State) {
  const res = await model.invoke(
    `Create a 3-point outline from:\n${state.research}`
  );
  return { outline: res.content as string };
}

async function writer(state: typeof PipelineState.State) {
  const res = await model.invoke(
    `Write a short article from this outline:\n${state.outline}`
  );
  return { article: res.content as string };
}

// ── 3. Graph ───────────────────────────────────────────────────────
const graph = new StateGraph(PipelineState)
  .addNode("researcher", researcher)
  .addNode("outliner", outliner)
  .addNode("writer", writer)
  .addEdge(START, "researcher")
  .addEdge("researcher", "outliner")
  .addEdge("outliner", "writer")
  .addEdge("writer", END)
  .compile();

// ── 4. Stream mode: "updates" ──────────────────────────────────────
// Shows ONLY what each node changed
console.log("=== Stream Mode: updates ===\n");

const updatesStream = await graph.stream(
  { topic: "The future of AI agents" },
  { streamMode: "updates" }
);

for await (const update of updatesStream) {
  // Each update is { nodeName: { field: value } }
  const [nodeName, changes] = Object.entries(update)[0];
  console.log(`📍 Node: ${nodeName}`);
  console.log(`   Changed fields: ${Object.keys(changes as object).join(", ")}`);
  console.log("─".repeat(50));
}

// ── 5. Stream mode: "values" ───────────────────────────────────────
// Shows the FULL state after each node
console.log("\n=== Stream Mode: values ===\n");

const valuesStream = await graph.stream(
  { topic: "Graph-based AI orchestration" },
  { streamMode: "values" }
);

let stepNum = 0;
for await (const fullState of valuesStream) {
  stepNum++;
  console.log(`📍 Step ${stepNum}:`);
  console.log(`   topic: ${fullState.topic ? "✓" : "✗"}`);
  console.log(`   research: ${fullState.research ? "✓" : "✗"}`);
  console.log(`   outline: ${fullState.outline ? "✓" : "✗"}`);
  console.log(`   article: ${fullState.article ? "✓" : "✗"}`);
  console.log("─".repeat(50));
}
