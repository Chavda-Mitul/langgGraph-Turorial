/**
 * 04-tool-integration/02-custom-tool-graph.ts
 * ─────────────────────────────────────────────
 * Custom Tool Graph: Build a research agent with web search tools.
 *
 * Key Concepts:
 * - Creating domain-specific tools
 * - Tool calling in a loop until the agent is satisfied
 * - Combining multiple tool results into a coherent response
 *
 * Run: npx ts-node --esm src/04-tool-integration/02-custom-tool-graph.ts
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

// ── 1. Domain-specific tools ───────────────────────────────────────
const searchKnowledgeBase = tool(
  async ({ query }) => {
    // Simulated knowledge base search
    const kb: Record<string, string> = {
      langgraph: "LangGraph is a framework for building stateful, multi-actor AI apps with LLMs. It uses directed graphs for orchestration and supports cycles, persistence, and human-in-the-loop.",
      langchain: "LangChain is a framework for developing applications powered by LLMs. It provides tools, chains, and agents for building AI workflows.",
      agents: "AI Agents are systems that use LLMs to decide which actions to take. They can use tools, maintain state, and adapt their behavior based on observations.",
      rag: "RAG (Retrieval-Augmented Generation) combines LLMs with external knowledge retrieval. Documents are embedded, stored in vector DBs, and retrieved at query time.",
    };

    const key = Object.keys(kb).find(k => query.toLowerCase().includes(k));
    return key ? kb[key] : `No results found for: ${query}`;
  },
  {
    name: "search_knowledge_base",
    description: "Search the internal knowledge base for AI/ML topics",
    schema: z.object({
      query: z.string().describe("Search query"),
    }),
  }
);

const generateQuiz = tool(
  async ({ topic, difficulty }) => {
    const quizzes: Record<string, string> = {
      easy: `Q: What does ${topic} stand for?\nA: [Answer based on context]`,
      medium: `Q: Explain how ${topic} differs from traditional approaches.\nA: [Requires understanding of fundamentals]`,
      hard: `Q: Design a system using ${topic} that handles failures gracefully.\nA: [Requires deep architectural knowledge]`,
    };
    return quizzes[difficulty] || quizzes["medium"];
  },
  {
    name: "generate_quiz",
    description: "Generate a quiz question about a topic at specified difficulty",
    schema: z.object({
      topic: z.string().describe("Topic for the quiz"),
      difficulty: z.enum(["easy", "medium", "hard"]).describe("Difficulty level"),
    }),
  }
);

const tools = [searchKnowledgeBase, generateQuiz];

// ── 2. Model + Graph ───────────────────────────────────────────────
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.3,
  maxTokens: 400,
}).bindTools(tools);

async function agent(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke([
    {
      role: "system",
      content: "You are a helpful AI tutor. Use the tools to search for information and create quizzes. Always search before answering questions about topics.",
    },
    ...state.messages,
  ]);
  return { messages: [response] };
}

function shouldContinue(state: typeof MessagesAnnotation.State): string {
  const lastMsg = state.messages[state.messages.length - 1] as AIMessage;
  return lastMsg.tool_calls?.length ? "tools" : END;
}

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", agent)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, ["tools", END])
  .addEdge("tools", "agent")
  .compile();

// ── 3. Test the research agent ─────────────────────────────────────
console.log("=== Custom Tool Graph: AI Tutor ===\n");

const result = await graph.invoke({
  messages: [
    new HumanMessage(
      "I want to learn about LangGraph. Search for info, then give me a medium difficulty quiz question."
    ),
  ],
});

console.log("Tutor:", result.messages[result.messages.length - 1].content);
