/**
 * 08-prebuilt-agents/02-agent-with-state.ts
 * ───────────────────────────────────────────
 * Agent with Custom State: Extend the prebuilt agent with extra fields.
 *
 * Key Concepts:
 * - createReactAgent can use MessagesAnnotation extended with custom fields
 * - stateModifier: Customize the system prompt dynamically from state
 * - Combine prebuilt convenience with custom state tracking
 *
 * Run: npx ts-node --esm src/08-prebuilt-agents/02-agent-with-state.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver, MessagesAnnotation, Annotation } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

// ── 1. Tools ───────────────────────────────────────────────────────
const searchDocs = tool(
  async ({ query }) => {
    const docs: Record<string, string> = {
      "state": "LangGraph state is managed via Annotations. Use Annotation.Root() to define your schema.",
      "nodes": "Nodes are functions: (state) => stateUpdate. They process state and return changes.",
      "edges": "Edges connect nodes. Use addEdge() for static, addConditionalEdges() for dynamic routing.",
      "tools": "Use ToolNode from @langchain/langgraph/prebuilt to auto-execute tool calls.",
    };
    const key = Object.keys(docs).find(k => query.toLowerCase().includes(k));
    return key ? docs[key] : `No docs found for: ${query}`;
  },
  {
    name: "search_docs",
    description: "Search LangGraph documentation",
    schema: z.object({
      query: z.string().describe("Search query"),
    }),
  }
);

// ── 2. Extended state ──────────────────────────────────────────────
const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
  username: Annotation<string>(),
  questionsAsked: Annotation<number>({
    reducer: (a, b) => a + b,
    default: () => 0,
  }),
});

// ── 3. Create agent with state modifier ────────────────────────────
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.5,
  maxTokens: 300,
});

const agent = createReactAgent({
  llm: model,
  tools: [searchDocs],
  // System prompt for the agent
  stateModifier: "You are a LangGraph tutor. Use the search_docs tool to answer technical questions. Be encouraging and educational.",
  checkpointSaver: new MemorySaver(),
});

// ── 4. Use the agent ───────────────────────────────────────────────
console.log("=== Agent with Custom State ===\n");

const config = { configurable: { thread_id: "tutor-session-1" } };

// Question 1
console.log("--- Q1 ---");
let result = await agent.invoke(
  {
    messages: [new HumanMessage("How do I define state in LangGraph?")],
  },
  config
);
console.log("Tutor:", result.messages[result.messages.length - 1].content);

// Question 2 — agent remembers context from Q1 via checkpointer
console.log("\n--- Q2 ---");
result = await agent.invoke(
  {
    messages: [new HumanMessage("What about edges? How do conditional edges work?")],
  },
  config
);
console.log("Tutor:", result.messages[result.messages.length - 1].content);
