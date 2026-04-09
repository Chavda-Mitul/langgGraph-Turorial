/**
 * 01-graph-basics/01-first-graph.ts
 * ──────────────────────────────────
 * Your First Graph: The simplest possible LangGraph workflow.
 *
 * Key Concepts:
 * - StateGraph: Container for nodes + edges
 * - Nodes: Functions that process state
 * - Edges: Connections between nodes (START → node → END)
 * - compile(): Turns the graph definition into an executable
 *
 * Think of it like a flowchart: each box is a node, each arrow is an edge.
 *
 * Run: npx ts-node --esm src/01-graph-basics/01-first-graph.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
} from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

// ── 1. Create a model ──────────────────────────────────────────────
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 256,
});

// ── 2. Define a node ─────────────────────────────────────────────── 
// A node is just a function: (state) => stateUpdate
// MessagesAnnotation gives us a `messages` array with a built-in
// reducer that appends new messages automatically.
async function chatNode(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);
  return { messages: [response] };  // This gets APPENDED to state.messages
}

// ── 3. Build the graph ─────────────────────────────────────────────
//
//   START ──→ chat ──→ END
// 
const graph = new StateGraph(MessagesAnnotation)
  .addNode("chat", chatNode)     // Register the node
  .addEdge(START, "chat")        // START flows into "chat"
  .addEdge("chat", END)          // "chat" flows into END
  .compile();                    // Lock it in — now it 's executable

// ── 4. Run the graph ───────────────────────────────────────────────
const result = await graph.invoke({
  messages: [new HumanMessage("Explain what a graph is in computer science, in 2 sentences.")],
});

console.log("=== First Graph ===");
console.log("Messages in state:", result.messages.length);
console.log("\nUser:", result.messages[0].content);
console.log("\nAI:", result.messages[result.messages.length - 1].content);
