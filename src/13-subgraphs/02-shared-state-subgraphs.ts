/**
 * 13-subgraphs/02-shared-state-subgraphs.ts
 * ───────────────────────────────────────────
 * Shared State Subgraphs: When parent and child share the SAME state.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THEORY: Two Styles of Subgraph Composition
 * ═══════════════════════════════════════════════════════════════════
 *
 * Style 1: DIFFERENT STATE (previous example)
 *   - Parent and child have different Annotation schemas
 *   - You manually map fields between them
 *   - Best for: reusable, independent subgraphs
 *
 * Style 2: SHARED STATE (this example)
 *   - Parent and child share the SAME Annotation schema
 *   - Child nodes directly read/write parent state
 *   - Best for: breaking a large graph into logical sections
 *   - Simpler but less encapsulated
 *
 * When to use which:
 *   Shared state → Internal decomposition (same team, same state)
 *   Different state → External composition (different teams, APIs)
 *
 * ═══════════════════════════════════════════════════════════════════
 *
 * Architecture:
 *   Parent Graph:
 *     START → intake → [validation_subgraph] → process → END
 *
 *   Validation Subgraph (shares parent state):
 *     START → check_format → check_length → END
 *
 * Run: npx ts-node --esm src/13-subgraphs/02-shared-state-subgraphs.ts
 */

import {
  Annotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

// ── Shared state — used by BOTH parent and child ──────────────────
const PipelineState = Annotation.Root({
  input: Annotation<string>(),
  isValid: Annotation<boolean>({
    reducer: (_, b) => b,
    default: () => false,
  }),
  errors: Annotation<string[]>({
    reducer: (existing, update) => [...existing, ...update],
    default: () => [],
  }),
  output: Annotation<string>(),
});

// ═══════════════════════════════════════════════════════════════════
// CHILD SUBGRAPH: Validation Pipeline
// ═══════════════════════════════════════════════════════════════════
// Uses the SAME PipelineState — reads `input`, writes `errors` and `isValid`.

function checkFormat(state: typeof PipelineState.State) {
  console.log("  ✅ [Validator] Checking format...");
  const errors: string[] = [];

  if (!state.input || state.input.trim().length === 0) {
    errors.push("Input cannot be empty");
  }
  if (state.input && !state.input.match(/^[a-zA-Z]/)) {
    errors.push("Input must start with a letter");
  }

  return { errors };
}

function checkLength(state: typeof PipelineState.State) {
  console.log("  ✅ [Validator] Checking length...");
  const errors: string[] = [];

  if (state.input && state.input.length < 5) {
    errors.push("Input must be at least 5 characters");
  }
  if (state.input && state.input.length > 500) {
    errors.push("Input must be under 500 characters");
  }

  // Set final validity — no errors across all checks
  const allErrors = [...state.errors, ...errors];
  return {
    errors,
    isValid: allErrors.length === 0,
  };
}

// Compile validation subgraph — uses SAME state as parent
const validationSubgraph = new StateGraph(PipelineState)
  .addNode("check_format", checkFormat)
  .addNode("check_length", checkLength)
  .addEdge(START, "check_format")
  .addEdge("check_format", "check_length")
  .addEdge("check_length", END)
  .compile();

// ═══════════════════════════════════════════════════════════════════
// PARENT GRAPH
// ═══════════════════════════════════════════════════════════════════

function intake(state: typeof PipelineState.State) {
  console.log("📥 [Parent] Intake:", state.input);
  return {};
}

function process(state: typeof PipelineState.State) {
  console.log("⚙️  [Parent] Processing...");
  if (!state.isValid) {
    return {
      output: `❌ Rejected. Errors: ${state.errors.join(", ")}`,
    };
  }
  return {
    output: `✅ Processed: "${state.input.toUpperCase()}"`,
  };
}

// Add the compiled subgraph directly as a node!
// Since they share state, no mapping is needed.
const parentGraph = new StateGraph(PipelineState)
  .addNode("intake", intake)
  .addNode("validate", validationSubgraph)  // Subgraph as a node!
  .addNode("process", process)
  .addEdge(START, "intake")
  .addEdge("intake", "validate")
  .addEdge("validate", "process")
  .addEdge("process", END)
  .compile();

// ═══════════════════════════════════════════════════════════════════
// Run with different inputs
// ═══════════════════════════════════════════════════════════════════

console.log("=== Shared State Subgraphs ===\n");

// Test 1: Valid input
console.log("--- Test 1: Valid Input ---");
let result = await parentGraph.invoke({
  input: "Hello, LangGraph!",
});
console.log("Output:", result.output);
console.log("Valid:", result.isValid);
console.log("Errors:", result.errors);

// Test 2: Too short
console.log("\n--- Test 2: Too Short ---");
result = await parentGraph.invoke({
  input: "Hi",
});
console.log("Output:", result.output);
console.log("Valid:", result.isValid);
console.log("Errors:", result.errors);

// Test 3: Starts with number
console.log("\n--- Test 3: Invalid Format ---");
result = await parentGraph.invoke({
  input: "123 invalid start",
});
console.log("Output:", result.output);
console.log("Valid:", result.isValid);
console.log("Errors:", result.errors);
