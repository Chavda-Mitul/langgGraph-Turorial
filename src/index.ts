/**
 * LangGraph Tutorial - Quick Demo
 * ────────────────────────────────
 * Run: npx ts-node --esm src/index.ts
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

// A simple single-node graph
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

console.log("🔗 LangGraph says:", result.messages[result.messages.length - 1].content);
