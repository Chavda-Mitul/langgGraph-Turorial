/**
 * 13-subgraphs/01-nested-graphs.ts
 * ──────────────────────────────────
 * Subgraphs: Compose graphs inside other graphs.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THEORY: Why Subgraphs?
 * ═══════════════════════════════════════════════════════════════════
 *
 * As your AI systems grow, a single flat graph becomes hard to manage.
 * Subgraphs let you:
 *
 * 1. ENCAPSULATION: Each subgraph manages its own internal state.
 *    The parent graph only sees input/output — not internal steps.
 *
 * 2. REUSABILITY: Build a "research" subgraph once, use it in
 *    multiple parent graphs (article writer, report generator, etc.)
 *
 * 3. TEAM COLLABORATION: Different team members can work on
 *    different subgraphs independently.
 *
 * 4. TESTING: Test each subgraph in isolation before composing.
 *
 * Think of it like functions in programming:
 *   - A node is like a single statement
 *   - A subgraph is like a function call (encapsulated logic)
 *   - The parent graph is like main()
 *
 * ═══════════════════════════════════════════════════════════════════
 *
 * Key Concepts:
 * - A compiled graph can be added as a node in another graph
 * - State mapping: parent state ↔ child state can differ
 * - The subgraph runs as a single "step" from the parent's perspective
 * - Subgraphs can have their own checkpointer for internal persistence
 *
 * Architecture:
 *
 *   Parent Graph:
 *   ┌──────────────────────────────────────────────────┐
 *   │  START → gather_input → [research_subgraph] → summarize → END │
 *   └──────────────────────────────────────────────────┘
 *                                │
 *                   Research Subgraph (internal):
 *                   ┌───────────────────────────┐
 *                   │ START → search → analyze → END │
 *                   └───────────────────────────┘
 *
 * Run: npx ts-node --esm src/13-subgraphs/01-nested-graphs.ts
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

// ═══════════════════════════════════════════════════════════════════
// STEP 1: Build the CHILD subgraph (Research Pipeline)
// ═══════════════════════════════════════════════════════════════════
// This subgraph has its OWN state — completely independent.

const ResearchState = Annotation.Root({
  query: Annotation<string>(),
  rawFindings: Annotation<string>(),
  analysis: Annotation<string>(),
});

async function search(state: typeof ResearchState.State) {
  console.log("    🔍 [Subgraph] Searching for:", state.query);
  const res = await model.invoke(
    `Research the following topic and provide 3 key facts: ${state.query}`
  );
  return { rawFindings: res.content as string };
}

async function analyze(state: typeof ResearchState.State) {
  console.log("    🧬 [Subgraph] Analyzing findings...");
  const res = await model.invoke(
    `Analyze these findings and extract the most important insight:\n\n${state.rawFindings}`
  );
  return { analysis: res.content as string };
}

// Compile the subgraph — this is a standalone, reusable unit
const researchSubgraph = new StateGraph(ResearchState)
  .addNode("search", search)
  .addNode("analyze", analyze)
  .addEdge(START, "search")
  .addEdge("search", "analyze")
  .addEdge("analyze", END)
  .compile();

// ═══════════════════════════════════════════════════════════════════
// STEP 2: Build the PARENT graph
// ═══════════════════════════════════════════════════════════════════
// The parent has DIFFERENT state fields. The subgraph is just a node.

const ParentState = Annotation.Root({
  topic: Annotation<string>(),
  researchResult: Annotation<string>(),
  finalSummary: Annotation<string>(),
});

// Gather input — prepares data for the subgraph
async function gatherInput(state: typeof ParentState.State) {
  console.log("📥 [Parent] Gathering input for research...");
  return {}; // topic is already in state
}

// Wrapper node that calls the subgraph
// This is where STATE MAPPING happens:
//   Parent's `topic` → Subgraph's `query`
//   Subgraph's `analysis` → Parent's `researchResult`
async function runResearch(state: typeof ParentState.State) {
  console.log("📚 [Parent] Invoking research subgraph...");

  // Invoke the subgraph with mapped state
  const result = await researchSubgraph.invoke({
    query: state.topic,  // Map parent field → child field
  });

  // Map child output → parent field
  return { researchResult: result.analysis };
}

// Summarize — uses research results
async function summarize(state: typeof ParentState.State) {
  console.log("📝 [Parent] Creating final summary...");
  const res = await model.invoke(
    `Write a concise executive summary based on this research:\n\n${state.researchResult}\n\nKeep it under 100 words.`
  );
  return { finalSummary: res.content as string };
}

// ═══════════════════════════════════════════════════════════════════
// STEP 3: Compose the parent graph
// ═══════════════════════════════════════════════════════════════════

const parentGraph = new StateGraph(ParentState)
  .addNode("gather_input", gatherInput)
  .addNode("research", runResearch)  // Subgraph runs here!
  .addNode("summarize", summarize)
  .addEdge(START, "gather_input")
  .addEdge("gather_input", "research")
  .addEdge("research", "summarize")
  .addEdge("summarize", END)
  .compile();

// ═══════════════════════════════════════════════════════════════════
// STEP 4: Run it
// ═══════════════════════════════════════════════════════════════════

console.log("=== Subgraphs: Nested Graph Composition ===\n");

const result = await parentGraph.invoke({
  topic: "How LangGraph uses directed graphs for AI agent orchestration",
});

console.log("\n" + "═".repeat(50));
console.log("📊 RESULTS");
console.log("═".repeat(50));
console.log("\nTopic:", result.topic);
console.log("\n📚 Research Result:\n", result.researchResult);
console.log("\n📝 Final Summary:\n", result.finalSummary);

// ═══════════════════════════════════════════════════════════════════
// BONUS: Demonstrate subgraph REUSABILITY
// ═══════════════════════════════════════════════════════════════════
// The same subgraph can be used independently!

console.log("\n\n=== Subgraph used independently ===\n");

const standaloneResult = await researchSubgraph.invoke({
  query: "TypeScript type safety in AI applications",
});

console.log("Standalone analysis:", standaloneResult.analysis);
