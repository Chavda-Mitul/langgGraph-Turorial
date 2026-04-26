/**
 * 06-human-in-the-loop/01-interrupt-approve.ts
 * ──────────────────────────────────────────────
 * Interrupt & Approve: Pause the graph for human approval.
 *
 * Key Concepts:
 * - interrupt(prompt): Pauses execution, returns prompt to caller
 * - Command({ resume: value }): Resumes with human-provided value
 * - interruptBefore: ["nodeName"] — interrupt BEFORE a node runs
 * - The graph "freezes" — checkpointer saves full state
 * - On resume, execution continues exactly where it left off
 *
 * Flow:
 *   START → planner → [INTERRUPT] → executor → END
 *                     (human approves)
 *
 * Run: npx ts-node --esm src/06-human-in-the-loop/01-interrupt-approve.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  Annotation,
  StateGraph,
  START,
  END,
  MemorySaver,
  interrupt,
  Command,
} from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.5,
  maxTokens: 300,
});

// ── 1. State ──
const ApprovalState = Annotation.Root({
  request: Annotation<string>(),
  plan: Annotation<string>(),
  approved: Annotation<boolean>({
    reducer: (_, b) => b,
    default: () => false,
  }),
  result: Annotation<string>(),
});

// ── 2. Planner creates a plan ──
async function planner(state: typeof ApprovalState.State) {
  console.log("📋 Planning:", state.request);
  const res = await model.invoke(
    `Create a short 3-step plan for: ${state.request}`
  );
  return { plan: res.content as string };
}

// ── 3. Human review node — pauses for approval ──
async function humanReview(state: typeof ApprovalState.State) {
  console.log("\n⏸️  Graph paused! Waiting for human approval...\n");

  // interrupt() freezes the graph and sends the prompt to the caller
  const decision = await interrupt(
    `Please review this plan:\n${state.plan}\n\nApprove? (yes/no)`
  );

  // When resumed, `decision` contains what the human provided
  const approved = decision === "yes" || decision === "approve";
  console.log(`\n✅ Human decision: ${approved ? "APPROVED" : "REJECTED"}`);
  return { approved };
}

// ── 4. Executor runs the plan ──
async function executor(state: typeof ApprovalState.State) {
  if (!state.approved) {
    return { result: "❌ Plan was rejected. No action taken." };
  }
  console.log("⚡ Executing approved plan...");
  const res = await model.invoke(
    `Execute this plan and describe what was accomplished:\n${state.plan}`
  );
  return { result: res.content as string };
}

// ── 5. Build graph ──
const checkpointer = new MemorySaver();

const graph = new StateGraph(ApprovalState)
  .addNode("planner", planner)
  .addNode("human_review", humanReview)
  .addNode("executor", executor)
  .addEdge(START, "planner")
  .addEdge("planner", "human_review")
  .addEdge("human_review", "executor")
  .addEdge("executor", END)
  .compile({ checkpointer });

// ── 6. Run with interrupt and resume ──
const config = { configurable: { thread_id: "approval-demo" } };

console.log("=== Human-in-the-Loop: Approval Workflow ===\n");

// Phase 1: Run until interrupt
console.log("--- Phase 1: Planning ---");
let result = await graph.invoke(
  { request: "Deploy a new version of the API to production" },
  config
);

// Check state — graph is paused at human_review
const state = await graph.getState(config);
console.log("Graph paused. Next nodes:", state.next);

// Phase 2: Resume with human approval
console.log("\n--- Phase 2: Human approves ---");
result = await graph.invoke(
  new Command({ resume: "yes" }),
  config
);

console.log("\n--- Final Result ---");
console.log("Approved:", result.approved);
console.log("Result:", result.result);
