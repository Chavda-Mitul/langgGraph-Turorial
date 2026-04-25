/**
 * 05-memory-checkpoints/01-memory-saver.ts
 * ──────────────────────────────────────────
 * MemorySaver: Persist conversation state across invocations.
 *
 * Key Concepts:
 * - MemorySaver: In-memory checkpointer (stores state snapshots)
 * - thread_id: Identifies a conversation (like a session ID)
 * - configurable: Pass thread_id via config to separate conversations
 * - State persists between invoke() calls for the same thread
 * - Different threads = completely separate conversations
 *
 * Run: npx ts-node --esm src/05-memory-checkpoints/01-memory-saver.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  MemorySaver,
} from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 200,
});

// ── 1. Build graph with checkpointer ───────────────────────────────
const graph = new StateGraph(MessagesAnnotation)
  .addNode("chat", async (state) => {
    const response = await model.invoke([
      { role: "system", content: "You are a friendly assistant. Remember what the user tells you." },
      ...state.messages,
    ]);
    return { messages: [response] };
  })
  .addEdge(START, "chat")
  .addEdge("chat", END)
  .compile({
    checkpointer: new MemorySaver(), // ← This enables persistence!
  });

// ── 2. Thread-based conversations ──────────────────────────────────
// Each thread_id is an independent conversation with its own memory.

const thread1 = { configurable: { thread_id: "user-mansi" } };
const thread2 = { configurable: { thread_id: "user-rahul" } };

console.log("=== Memory Saver: Thread-based Conversations ===\n");

// Conversation 1: Thread 1
console.log("--- Thread 1: Mansi ---");
let result = await graph.invoke(
  { messages: [new HumanMessage("Hi! My name is Mansi and I love TypeScript.")] },
  thread1
);
console.log("AI:", result.messages[result.messages.length - 1].content);

// Conversation 2: Thread 2 (separate!)
console.log("\n--- Thread 2: Rahul ---");
result = await graph.invoke(
  { messages: [new HumanMessage("Hey, I'm Rahul. I'm learning Python.")] },
  thread2
);
console.log("AI:", result.messages[result.messages.length - 1].content);

// Continue Thread 1 — it remembers Mansi!
console.log("\n--- Thread 1: Mansi (continued) ---");
result = await graph.invoke(
  { messages: [new HumanMessage("What's my name and what language do I like?")] },
  thread1
);
console.log("AI:", result.messages[result.messages.length - 1].content);

// Continue Thread 2 — it remembers Rahul!
console.log("\n--- Thread 2: Rahul (continued) ---");
result = await graph.invoke(
  { messages: [new HumanMessage("What's my name and what am I learning?")] },
  thread2
);
console.log("AI:", result.messages[result.messages.length - 1].content);

// Check state — how many messages are in each thread?
console.log("\n--- State Inspection ---");
const state1 = await graph.getState(thread1);
const state2 = await graph.getState(thread2);
console.log(`Thread 1 messages: ${state1.values.messages.length}`);
console.log(`Thread 2 messages: ${state2.values.messages.length}`);

