/**
 * 03-conditional-edges/02-loops.ts
 * ─────────────────────────────────
 * Loops: Graphs that cycle back — the core of agentic behavior.
 *
 * Key Concepts:
 * - Conditional edges can point BACK to earlier nodes (loops!)
 * - This is what makes agents "think → act → observe → think again"
 * - Use a condition to break the loop (otherwise infinite!)
 * - This pattern is the foundation of ReAct agents
 *
 * Flow:
 *   START → writer → reviewer ──→ (good enough?) ──→ END
 *                       ↑              │ (no)
 *                       └──────────────┘
 *
 * Run: npx ts-node --esm src/03-conditional-edges/02-loops.ts
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

// ── 1. State ───────────────────────────────────────────────────────
const WritingState = Annotation.Root({
  topic: Annotation<string>(),
  draft: Annotation<string>(),
  feedback: Annotation<string>(),
  revision: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),
  approved: Annotation<boolean>({
    reducer: (_, b) => b,
    default: () => false,
  }),
});

// ── 2. Writer node ─────────────────────────────────────────────────
async function writer(state: typeof WritingState.State) {
  const revision = state.revision + 1;
  console.log(`\n✍️  Writer: Revision #${revision}`);

  let prompt: string;
  if (state.revision === 0) {
    prompt = `Write a short paragraph (3-4 sentences) about: ${state.topic}`;
  } else {
    prompt = `Improve this draft based on feedback.\n\nDraft:\n${state.draft}\n\nFeedback:\n${state.feedback}\n\nWrite an improved version (3-4 sentences).`;
  }

  const res = await model.invoke(prompt);
  return { draft: res.content as string, revision };
}

// ── 3. Reviewer node ───────────────────────────────────────────────
async function reviewer(state: typeof WritingState.State) {
  console.log("📝 Reviewer: Evaluating draft...");

  const res = await model.invoke(
    `Review this draft. If it's good, say exactly "APPROVED". Otherwise give ONE specific improvement suggestion.\n\nDraft:\n${state.draft}`
  );

  const feedback = res.content as string;
  const approved = feedback.toUpperCase().includes("APPROVED");

  console.log(`   Verdict: ${approved ? "✅ Approved!" : "🔄 Needs revision"}`);
  return { feedback, approved };
}

// ── 4. Loop condition ──────────────────────────────────────────────
function shouldContinue(state: typeof WritingState.State): string {
  // Stop if approved OR we've done 3 revisions (safety limit)
  if (state.approved || state.revision >= 3) {
    return END;
  }
  return "writer"; // Loop back!
}

// ── 5. Build graph with loop ───────────────────────────────────────
const graph = new StateGraph(WritingState)
  .addNode("writer", writer)
  .addNode("reviewer", reviewer)
  .addEdge(START, "writer")
  .addEdge("writer", "reviewer")
  .addConditionalEdges("reviewer", shouldContinue, ["writer", END])
  .compile();

// ── 6. Run ─────────────────────────────────────────────────────────
console.log("=== Loop-based Writing Agent ===");

const result = await graph.invoke({
  topic: "Why graph-based orchestration is the future of AI agents",
});

console.log("\n=== Final Result ===");
console.log(`Revisions: ${result.revision}`);
console.log(`Approved: ${result.approved}`);
console.log(`\nFinal Draft:\n${result.draft}`);
