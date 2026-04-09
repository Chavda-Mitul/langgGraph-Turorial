/**
 * 16-visualization/01-graph-visualization.ts
 * ────────────────────────────────────────────
 * Graph Visualization: See your graph structure as a diagram.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THEORY: Why Visualize Your Graphs?
 * ═══════════════════════════════════════════════════════════════════
 *
 * When your graph has 5+ nodes, conditional edges, and loops,
 * it becomes HARD to reason about the flow just from code.
 *
 * Visualization helps you:
 * 1. VERIFY: "Does the flow match what I intended?"
 * 2. DEBUG: "Why is the agent going to node X instead of Y?"
 * 3. COMMUNICATE: Share architecture with team members
 * 4. DOCUMENT: Auto-generated diagrams stay in sync with code
 *
 * LangGraph can generate:
 * - Mermaid diagrams (text-based, renders in GitHub/VS Code/Notion)
 * - ASCII representations (for terminal output)
 * - JSON graph structure (for custom rendering)
 *
 * Mermaid Syntax Crash Course:
 *   graph TD       → Top-Down direction
 *   A --> B        → Edge from A to B
 *   A -->|label| B → Edge with a label
 *   A{decision}    → Diamond shape (conditional)
 *   A[process]     → Rectangle shape (node)
 *   A([start])     → Rounded rectangle (start/end)
 *
 * You can paste Mermaid output into:
 *   - GitHub README/issues (```mermaid code blocks)
 *   - https://mermaid.live (online editor)
 *   - VS Code with Mermaid extension
 *   - Notion, Obsidian, and many other tools
 *
 * ═══════════════════════════════════════════════════════════════════
 *
 * Run: npx ts-node --esm src/16-visualization/01-graph-visualization.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  Annotation,
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  MemorySaver,
} from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// ══════════════════════════════════════════════════════════════════
// EXAMPLE 1: Simple Graph Visualization
// ══════════════════════════════════════════════════════════════════

const SimpleState = Annotation.Root({
  input: Annotation<string>(),
  output: Annotation<string>(),
});

const simpleGraph = new StateGraph(SimpleState)
  .addNode("process", async (state) => ({
    output: state.input.toUpperCase(),
  }))
  .addEdge(START, "process")
  .addEdge("process", END)
  .compile();

console.log("=== Graph Visualization ===\n");
console.log("─── Example 1: Simple Graph ───\n");

// getGraph() returns the graph structure
const simpleGraphDef = simpleGraph.getGraph();

// drawMermaid() generates a Mermaid diagram string
const simpleMermaid = simpleGraphDef.drawMermaid();
console.log("Mermaid Diagram:");
console.log(simpleMermaid);

// ══════════════════════════════════════════════════════════════════
// EXAMPLE 2: Complex Agent Graph with Tools
// ══════════════════════════════════════════════════════════════════

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  maxTokens: 200,
});

const searchTool = tool(
  async ({ query }) => `Results for: ${query}`,
  {
    name: "search",
    description: "Search for information",
    schema: z.object({ query: z.string() }),
  }
);

const tools = [searchTool];
const modelWithTools = model.bindTools(tools);

function shouldContinue(state: typeof MessagesAnnotation.State): string {
  const lastMsg = state.messages[state.messages.length - 1] as AIMessage;
  return lastMsg.tool_calls?.length ? "tools" : END;
}

const agentGraph = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state) => {
    const response = await modelWithTools.invoke(state.messages);
    return { messages: [response] };
  })
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, ["tools", END])
  .addEdge("tools", "agent")
  .compile();

console.log("\n─── Example 2: Agent with Tools ───\n");
const agentMermaid = agentGraph.getGraph().drawMermaid();
console.log("Mermaid Diagram:");
console.log(agentMermaid);

// ══════════════════════════════════════════════════════════════════
// EXAMPLE 3: Multi-step Pipeline with Conditional Routing
// ══════════════════════════════════════════════════════════════════

const PipelineState = Annotation.Root({
  input: Annotation<string>(),
  category: Annotation<string>(),
  result: Annotation<string>(),
});

function classify(state: typeof PipelineState.State) {
  const category = state.input.length > 20 ? "complex" : "simple";
  return { category };
}

function handleSimple(state: typeof PipelineState.State) {
  return { result: "Quick answer: " + state.input };
}

function handleComplex(state: typeof PipelineState.State) {
  return { result: "Detailed analysis of: " + state.input };
}

function routeByCategory(state: typeof PipelineState.State): string {
  return state.category === "complex" ? "handle_complex" : "handle_simple";
}

const pipelineGraph = new StateGraph(PipelineState)
  .addNode("classify", classify)
  .addNode("handle_simple", handleSimple)
  .addNode("handle_complex", handleComplex)
  .addEdge(START, "classify")
  .addConditionalEdges("classify", routeByCategory, [
    "handle_simple",
    "handle_complex",
  ])
  .addEdge("handle_simple", END)
  .addEdge("handle_complex", END)
  .compile();

console.log("\n─── Example 3: Conditional Pipeline ───\n");
const pipelineMermaid = pipelineGraph.getGraph().drawMermaid();
console.log("Mermaid Diagram:");
console.log(pipelineMermaid);

// ══════════════════════════════════════════════════════════════════
// EXAMPLE 4: Inspecting Graph Structure (JSON)
// ══════════════════════════════════════════════════════════════════

console.log("\n─── Example 4: Graph Structure (JSON) ───\n");

const graphDef = pipelineGraph.getGraph();

// Get all nodes
console.log("Nodes:");
for (const [id, node] of Object.entries(graphDef.nodes)) {
  console.log(`  - ${id}: ${node.constructor.name}`);
}

// Get all edges
console.log("\nEdges:");
for (const edge of graphDef.edges) {
  const conditional = edge.conditional ? " (conditional)" : "";
  console.log(`  ${edge.source} → ${edge.target}${conditional}`);
}

// ══════════════════════════════════════════════════════════════════
// TIP: How to use the Mermaid output
// ══════════════════════════════════════════════════════════════════

console.log("\n─── How to Use Mermaid Output ───\n");
console.log("1. Copy the Mermaid diagram above");
console.log("2. Paste into one of these:");
console.log("   - GitHub: Use ```mermaid code block in README/issues");
console.log("   - Online: https://mermaid.live");
console.log("   - VS Code: Install 'Mermaid Preview' extension");
console.log("   - Notion/Obsidian: Built-in Mermaid support");
console.log("\n3. Or save to a file:");
console.log('   import fs from "fs";');
console.log('   fs.writeFileSync("graph.mmd", mermaidString);');
