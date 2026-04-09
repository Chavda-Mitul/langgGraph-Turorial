/**
 * 03-conditional-edges/01-basic-routing.ts
 * ─────────────────────────────────────────
 * Conditional Edges: Dynamic routing based on state.
 *
 * Key Concepts:
 * - addConditionalEdges(sourceNode, routerFn, possibleDestinations)
 * - Router function: (state) => "nodeName" | "__end__"
 * - The router function is pure logic — it decides where to go next
 * - This is the foundation of agent decision-making
 *
 * Flow:
 *                 ┌──→ positive_response ──→ END
 *   START → classifier ──→ negative_response ──→ END
 *                 └──→ neutral_response  ──→ END
 *
 * Run: npx ts-node --esm src/03-conditional-edges/01-basic-routing.ts
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
  temperature: 0.3,
  maxTokens: 150,
});

// ── 1. State ───────────────────────────────────────────────────────
const SentimentState = Annotation.Root({
  message: Annotation<string>(),
  sentiment: Annotation<string>(),
  response: Annotation<string>(),
});

// ── 2. Classifier node ─────────────────────────────────────────────
async function classifier(state: typeof SentimentState.State) {
  const res = await model.invoke(
    `Classify the sentiment of this message as exactly one word — "positive", "negative", or "neutral":\n\n"${state.message}"`
  );
  const sentiment = (res.content as string).toLowerCase().trim();
  console.log(`🔍 Classified as: ${sentiment}`);
  return { sentiment };
}

// ── 3. Response nodes ──────────────────────────────────────────────
async function positiveResponse(state: typeof SentimentState.State) {
  console.log("😊 Routing to positive handler");
  return { response: `Great to hear! "${state.message}" — that's wonderful!` };
}

async function negativeResponse(state: typeof SentimentState.State) {
  console.log("😔 Routing to negative handler");
  return { response: `I'm sorry about that. "${state.message}" — let me help.` };
}

async function neutralResponse(state: typeof SentimentState.State) {
  console.log("😐 Routing to neutral handler");
  return { response: `Understood. "${state.message}" — noted!` };
}

// ── 4. Router function ─────────────────────────────────────────────
// This is the KEY part — it reads state and returns a node name.
function routeBySentiment(state: typeof SentimentState.State): string {
  if (state.sentiment.includes("positive")) return "positive_response";
  if (state.sentiment.includes("negative")) return "negative_response";
  return "neutral_response";
}

// ── 5. Build graph with conditional edges ──────────────────────────
const graph = new StateGraph(SentimentState)
  .addNode("classifier", classifier)
  .addNode("positive_response", positiveResponse)
  .addNode("negative_response", negativeResponse)
  .addNode("neutral_response", neutralResponse)
  .addEdge(START, "classifier")
  // After classifier, the router decides which response node to visit
  .addConditionalEdges("classifier", routeBySentiment, [
    "positive_response",
    "negative_response",
    "neutral_response",
  ])
  .addEdge("positive_response", END)
  .addEdge("negative_response", END)
  .addEdge("neutral_response", END)
  .compile();

// ── 6. Test with different sentiments ──────────────────────────────
console.log("=== Conditional Routing ===\n");

const messages = [
  "I love learning about AI! It's so exciting!",
  "This bug has been frustrating me all day.",
  "The meeting is at 3pm tomorrow.",
];

for (const msg of messages) {
  console.log(`\nInput: "${msg}"`);
  const result = await graph.invoke({ message: msg });
  console.log(`Output: ${result.response}`);
  console.log("─".repeat(50));
}
