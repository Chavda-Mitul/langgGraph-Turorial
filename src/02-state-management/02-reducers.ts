/**
 * 02-state-management/02-reducers.ts
 * ───────────────────────────────────
 * Reducers: Control how state updates merge.
 *
 * Key Concepts:
 * - reducer: (existing, update) => merged — defines merge strategy
 * - default: () => initialValue — sets initial state
 * - Common patterns: overwrite, append, count, merge objects
 * - Without reducer: LAST WRITE WINS (overwrite)
 * - With reducer: YOU control the merge logic
 *
 * Run: npx ts-node --esm src/02-state-management/02-reducers.ts
 */

import {
  Annotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

// ── 1. Append reducer (accumulate items) ───────────────────────────
const LogState = Annotation.Root({
  // Every update appends to the array (like messages)
  logs: Annotation<string[]>({
    reducer: (existing, update) => [...existing, ...update],
    default: () => [],
  }),
  // Counter: each update adds to the total
  stepCount: Annotation<number>({
    reducer: (existing, update) => existing + update,
    default: () => 0,
  }),
});

const logGraph = new StateGraph(LogState)
  .addNode("step1", (_state) => {
    return {
      logs: ["Step 1: Initialized"],
      stepCount: 1,
    };
  })
  .addNode("step2", (_state) => {
    return {
      logs: ["Step 2: Processed"],
      stepCount: 1,
    };
  })
  .addNode("step3", (state) => {
    return {
      logs: [`Step 3: Done (total steps: ${state.stepCount + 1})`],
      stepCount: 1,
    };
  })
  .addEdge(START, "step1")
  .addEdge("step1", "step2")
  .addEdge("step2", "step3")
  .addEdge("step3", END)
  .compile();

const logResult = await logGraph.invoke({});

console.log("=== Append Reducer ===");
console.log("Logs:", logResult.logs);
console.log("Total steps:", logResult.stepCount);

// ── 2. Object merge reducer (deep merge) ──────────────────────────
const ProfileState = Annotation.Root({
  profile: Annotation<Record<string, string>>({
    reducer: (existing, update) => ({ ...existing, ...update }),
    default: () => ({}),
  }),
});

const profileGraph = new StateGraph(ProfileState)
  .addNode("set_name", () => ({
    profile: { name: "Mansi", role: "Developer" },
  }))
  .addNode("set_skills", () => ({
    profile: { language: "TypeScript", framework: "LangGraph" },
  }))
  .addEdge(START, "set_name")
  .addEdge("set_name", "set_skills")
  .addEdge("set_skills", END)
  .compile();

const profileResult = await profileGraph.invoke({});

console.log("\n=== Object Merge Reducer ===");
console.log("Profile:", profileResult.profile);
// { name: "Mansi", role: "Developer", language: "TypeScript", framework: "LangGraph" }

// ── 3. Conditional reducer (keep max) ──────────────────────────────
const ScoreState = Annotation.Root({
  scores: Annotation<number[]>({
    reducer: (existing, update) => [...existing, ...update],
    default: () => [],
  }),
  highScore: Annotation<number>({
    reducer: (existing, update) => Math.max(existing, update),
    default: () => 0,
  }),
});

const scoreGraph = new StateGraph(ScoreState)
  .addNode("round1", () => ({ scores: [75], highScore: 75 }))
  .addNode("round2", () => ({ scores: [92], highScore: 92 }))
  .addNode("round3", () => ({ scores: [88], highScore: 88 }))
  .addEdge(START, "round1")
  .addEdge("round1", "round2")
  .addEdge("round2", "round3")
  .addEdge("round3", END)
  .compile();

const scoreResult = await scoreGraph.invoke({});

console.log("\n=== Max Reducer ===");
console.log("All scores:", scoreResult.scores);  // [75, 92, 88]
console.log("High score:", scoreResult.highScore); // 92
