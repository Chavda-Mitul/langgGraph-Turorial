/**
 * 08-prebuilt-agents/02-agent-with-state.ts
 * ───────────────────────────────────────────
 * Agent with Custom State: Extend the prebuilt agent with extra fields.
 *
 * Key Concepts:
 * - createReactAgent can use MessagesAnnotation extended with custom fields
 * - `prompt` as a FUNCTION: dynamically build system prompt from state
 *   (replaces older `stateModifier` — `prompt` is the modern approach)
 * - `stateSchema`: Tell the agent about your custom state shape
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
import { HumanMessage, type BaseMessageLike } from "@langchain/core/messages";

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

// ── 3. Create agent with DYNAMIC prompt (function-based) ──────────
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.5,
  maxTokens: 300,
});

// The `prompt` parameter as a function receives state and returns messages.
// This is more powerful than a static string — you can personalize responses.
const agent = createReactAgent({
  llm: model,
  tools: [searchDocs],
  // `prompt` as a function: build system prompt dynamically from state
  // The function receives the full state (including custom fields) and
  // returns the messages array to send to the LLM.
  prompt: (state) => {
    // Access custom state fields — TypeScript infers the type from stateSchema
    const name = (state as any).username || "learner";
    const qCount = (state as any).questionsAsked || 0;
    return [
      {
        role: "system" as const,
        content: `You are a LangGraph tutor helping ${name}. ` +
          `They have asked ${qCount} question(s) so far. ` +
          `Use the search_docs tool to answer technical questions. Be encouraging and educational.`,
      },
      ...state.messages,
    ];
  },
  // `stateSchema` tells the agent about our custom state shape
  stateSchema: AgentState,
  checkpointSaver: new MemorySaver(),
});

// ── 4. Use the agent ───────────────────────────────────────────────
console.log("=== Agent with Custom State ===\n");

const config = { configurable: { thread_id: "tutor-session-1" } };

// Question 1 — passing custom state fields
console.log("--- Q1 ---");
let result = await agent.invoke(
  {
    messages: [new HumanMessage("How do I define state in LangGraph?")],
    username: "Mansi",
    questionsAsked: 1,
  },
  config
);
console.log("Tutor:", result.messages[result.messages.length - 1].content);

// Question 2 — agent remembers context from Q1 via checkpointer
// Note: questionsAsked uses a REDUCER (a + b), so it accumulates!
console.log("\n--- Q2 ---");
result = await agent.invoke(
  {
    messages: [new HumanMessage("What about edges? How do conditional edges work?")],
    questionsAsked: 1,  // Adds to existing count (reducer: a + b = 2)
  },
  config
);
console.log("Tutor:", result.messages[result.messages.length - 1].content);
