/**
 * 01-graph-basics/02-multi-node-graph.ts
 * ───────────────────────────────────────
 * Multi-Node Graph: Chain multiple processing steps as a graph.
 *
 * Key Concepts:
 * - Multiple nodes executing in sequence
 * - Each node reads state and returns updates
 * - State flows through the graph like water through pipes
 *
 * Flow: START → researcher → analyzer → END
 *
 * Run: npx ts-node --esm src/01-graph-basics/02-multi-node-graph.ts
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

// ── 1. Define custom state ─────────────────────────────────────────
// Unlike MessagesAnnotation (just messages), we define our OWN state.
// Each field can have a default value and a reducer.
const ResearchState = Annotation.Root({
  topic: Annotation<string>(),              // Input topic
  research: Annotation<string>(),           // Filled by researcher node
  analysis: Annotation<string>(),           // Filled by analyzer node
});

// ── 2. Researcher node ─────────────────────────────────────────────
async function researcher(state: typeof ResearchState.State) {
  console.log("📚 Researcher: Investigating", state.topic);

  const response = await model.invoke(
    `Research the following topic and provide key facts: ${state.topic}`
  );

  return { research: response.content as string };
}

// ── 3. Analyzer node ───────────────────────────────────────────────
async function analyzer(state: typeof ResearchState.State) {
  console.log("🔬 Analyzer: Analyzing research...");

  const response = await model.invoke(
    `Analyze this research and provide 3 key insights:\n\n${state.research}`
  );

  return { analysis: response.content as string };
}

// ── 4. Build the sequential graph ──────────────────────────────────
//
//   START ──→ researcher ──→ analyzer ──→ END
//
const graph = new StateGraph(ResearchState)
  .addNode("researcher", researcher)
  .addNode("analyzer", analyzer)
  .addEdge(START, "researcher")
  .addEdge("researcher", "analyzer")
  .addEdge("analyzer", END)
  .compile();

// ── 5. Run it ──────────────────────────────────────────────────────
const result = await graph.invoke({
  topic: "LangGraph and its role in AI agent development",
});

console.log("\n=== Multi-Node Graph Result ===");
console.log("Topic:", result.topic);
console.log("\n📚 Research:\n", result.research);
console.log("\n🔬 Analysis:\n", result.analysis);
