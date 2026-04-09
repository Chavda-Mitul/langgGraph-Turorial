/**
 * 05-memory-checkpoints/02-state-inspection.ts
 * ──────────────────────────────────────────────
 * State Inspection: Read and manipulate graph state.
 *
 * Key Concepts:
 * - getState(): Read the current state of a thread
 * - getStateHistory(): Get all historical states (time travel!)
 * - updateState(): Modify state externally (useful for corrections)
 * - State snapshots include: values, next nodes, config, metadata
 *
 * Run: npx ts-node --esm src/05-memory-checkpoints/02-state-inspection.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  Annotation,
  StateGraph,
  START,
  END,
  MemorySaver,
} from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.3,
  maxTokens: 200,
});

// ── 1. State with trackable fields ─────────────────────────────────
const TaskState = Annotation.Root({
  task: Annotation<string>(),
  steps: Annotation<string[]>({
    reducer: (existing, update) => [...existing, ...update],
    default: () => [],
  }),
  status: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "pending",
  }),
  result: Annotation<string>(),
});

// ── 2. Multi-step workflow ─────────────────────────────────────────
const graph = new StateGraph(TaskState)
  .addNode("plan", async (state) => {
    console.log("📋 Planning...");
    const res = await model.invoke(
      `Create a 3-step plan for: ${state.task}. Return only the steps, numbered.`
    );
    return {
      steps: ["Plan created"],
      status: "planning",
      result: res.content as string,
    };
  })
  .addNode("execute", async (state) => {
    console.log("⚡ Executing...");
    return {
      steps: ["Execution started", "Execution completed"],
      status: "done",
    };
  })
  .addEdge(START, "plan")
  .addEdge("plan", "execute")
  .addEdge("execute", END)
  .compile({
    checkpointer: new MemorySaver(),
  });

const config = { configurable: { thread_id: "task-001" } };

// ── 3. Run and inspect state ───────────────────────────────────────
console.log("=== State Inspection ===\n");

const result = await graph.invoke(
  { task: "Build an AI-powered code reviewer" },
  config
);

// Get current state
const currentState = await graph.getState(config);
console.log("\n--- Current State ---");
console.log("Status:", currentState.values.status);
console.log("Steps:", currentState.values.steps);
console.log("Next nodes:", currentState.next); // Empty — graph is done

// Get state history (time travel!)
console.log("\n--- State History (Time Travel) ---");
const history = graph.getStateHistory(config);
let snapshotCount = 0;

for await (const snapshot of history) {
  snapshotCount++;
  console.log(`\nSnapshot #${snapshotCount}:`);
  console.log("  Status:", snapshot.values.status);
  console.log("  Steps:", snapshot.values.steps);
  console.log("  Next:", snapshot.next);

  if (snapshotCount >= 4) break; // Limit output
}

// ── 4. Update state externally ─────────────────────────────────────
console.log("\n--- External State Update ---");
console.log("Before:", (await graph.getState(config)).values.status);

await graph.updateState(config, { status: "reviewed" });

console.log("After:", (await graph.getState(config)).values.status);
