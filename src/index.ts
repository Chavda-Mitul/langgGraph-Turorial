/**
 * LangGraph Tutorial — Quick Demo & Module Index
 * ─────────────────────────────────────────────────
 * This file runs a quick sanity check and lists all tutorial modules.
 *
 * Run: npx ts-node --esm src/index.ts
 *
 * To run individual modules:
 *   npx ts-node --esm src/01-graph-basics/01-first-graph.ts
 *   npx ts-node --esm src/09-mini-project/app.ts
 *   ... etc.
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 100,
});

// A simple single-node graph to verify setup
const graph = new StateGraph(MessagesAnnotation)
  .addNode("chat", async (state) => {
    const response = await model.invoke(state.messages);
    return { messages: [response] };
  })
  .addEdge(START, "chat")
  .addEdge("chat", END)
  .compile();

const result = await graph.invoke({
  messages: [new HumanMessage("What is LangGraph in one sentence?")],
});

console.log("LangGraph says:", result.messages[result.messages.length - 1].content);

console.log(`
╔══════════════════════════════════════════════════════════════╗
║              LangGraph Tutorial — Module Index               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  BEGINNER                                                    ║
║  01-graph-basics/        Nodes, edges, parallel execution    ║
║  02-state-management/    Annotations, reducers               ║
║  03-conditional-edges/   Dynamic routing, loops              ║
║                                                              ║
║  INTERMEDIATE                                                ║
║  04-tool-integration/    ToolNode, ReAct pattern             ║
║  05-memory-checkpoints/  MemorySaver, state inspection       ║
║  06-human-in-the-loop/   interrupt(), Command({ resume })    ║
║  07-streaming/           Stream modes, token streaming       ║
║                                                              ║
║  ADVANCED                                                    ║
║  08-prebuilt-agents/     createReactAgent, custom state      ║
║  09-mini-project/        Multi-agent content pipeline        ║
║  10-functional-api/      entrypoint + task                   ║
║  11-command-routing/     Command routing, agent handoffs     ║
║  12-long-term-memory/    InMemoryStore                       ║
║                                                              ║
║  PRODUCTION-READY                                            ║
║  13-subgraphs/           Nested graphs, composition          ║
║  14-error-handling/      Retry, fallback, resilience         ║
║  15-message-trimming/    Context window management           ║
║  16-visualization/       Mermaid diagrams, debugging         ║
║                                                              ║
║  Run any module:                                             ║
║  npx ts-node --esm src/<module>/<file>.ts                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
`);
