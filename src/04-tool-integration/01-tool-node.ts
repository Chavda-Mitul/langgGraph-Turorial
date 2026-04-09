/**
 * 04-tool-integration/01-tool-node.ts
 * ────────────────────────────────────
 * ToolNode: Let your graph call tools automatically.
 *
 * Key Concepts:
 * - tool(): Create typed tools with Zod schemas
 * - model.bindTools(): Tell the LLM about available tools
 * - ToolNode: Prebuilt node that executes tool calls from AI messages
 * - The agent loop: model → check tool_calls → execute → back to model
 *
 * This is the classic ReAct pattern implemented as a graph:
 *
 *   START → agent → (has tool calls?) → tools → agent → ... → END
 *
 * Run: npx ts-node --esm src/04-tool-integration/01-tool-node.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// ── 1. Define tools ────────────────────────────────────────────────
const calculator = tool(
  async ({ operation, a, b }) => {
    switch (operation) {
      case "add": return `${a} + ${b} = ${a + b}`;
      case "subtract": return `${a} - ${b} = ${a - b}`;
      case "multiply": return `${a} * ${b} = ${a * b}`;
      case "divide": return b !== 0 ? `${a} / ${b} = ${a / b}` : "Error: division by zero";
      default: return "Unknown operation";
    }
  },
  {
    name: "calculator",
    description: "Perform basic arithmetic operations",
    schema: z.object({
      operation: z.enum(["add", "subtract", "multiply", "divide"]),
      a: z.number().describe("First number"),
      b: z.number().describe("Second number"),
    }),
  }
);

const getCurrentTime = tool(
  async () => {
    return `Current time: ${new Date().toLocaleTimeString()}`;
  },
  {
    name: "get_current_time",
    description: "Get the current time",
    schema: z.object({}),
  }
);

const tools = [calculator, getCurrentTime];

// ── 2. Create model with tools ─────────────────────────────────────
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  maxTokens: 256,
});

// bindTools tells the LLM the tool schemas so it can generate tool_calls
const modelWithTools = model.bindTools(tools);

// ── 3. Define nodes ────────────────────────────────────────────────
// Agent node: calls the LLM
async function agentNode(state: typeof MessagesAnnotation.State) {
  console.log("🤖 Agent thinking...");
  const response = await modelWithTools.invoke(state.messages);
  return { messages: [response] };
}

// Tool node: executes any tool_calls from the AI message
const toolNode = new ToolNode(tools);

// ── 4. Router: should we call tools or end? ────────────────────────
function shouldCallTools(state: typeof MessagesAnnotation.State): string {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;

  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    console.log(`🔧 Tool calls: ${lastMessage.tool_calls.map(tc => tc.name).join(", ")}`);
    return "tools";
  }

  return "__end__";
}

// ── 5. Build the agent graph ───────────────────────────────────────
//
//   START → agent ──→ (tool_calls?) ──→ tools → agent → ... → END
//                          │ (no)
//                          └──→ END
//
const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", agentNode)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldCallTools, ["tools", "__end__"])
  .addEdge("tools", "agent")  // After tools, go back to agent
  .compile();

// ── 6. Run it ──────────────────────────────────────────────────────
console.log("=== Tool-Calling Agent ===\n");

// Test 1: Tool usage
console.log("--- Query 1: Math ---");
const result1 = await graph.invoke({
  messages: [new HumanMessage("What is 42 multiplied by 17?")],
});
console.log("Answer:", result1.messages[result1.messages.length - 1].content);

// Test 2: Multiple tools
console.log("\n--- Query 2: Multiple tools ---");
const result2 = await graph.invoke({
  messages: [new HumanMessage("What time is it, and what is 100 divided by 7?")],
});
console.log("Answer:", result2.messages[result2.messages.length - 1].content);

// Test 3: No tools needed
console.log("\n--- Query 3: No tools needed ---");
const result3 = await graph.invoke({
  messages: [new HumanMessage("What is LangGraph?")],
});
console.log("Answer:", result3.messages[result3.messages.length - 1].content);
