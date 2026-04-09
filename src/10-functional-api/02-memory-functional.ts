/**
 * 10-functional-api/02-memory-functional.ts
 * ───────────────────────────────────────────
 * Functional API Memory: getPreviousState + entrypoint.final
 *
 * Key Concepts:
 * - getPreviousState(): Access the saved state from the LAST invocation
 * - entrypoint.final(): Control what is RETURNED vs what is SAVED
 *   - value: what the caller receives
 *   - save: what gets persisted for next invocation
 * - This replaces MemorySaver + MessagesAnnotation for functional workflows
 * - Thread-based: each thread_id has its own memory
 *
 * Run: npx ts-node --esm src/10-functional-api/02-memory-functional.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import type { BaseMessage, BaseMessageLike } from "@langchain/core/messages";
import {
  addMessages,
  entrypoint,
  task,
  getPreviousState,
  MemorySaver,
} from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 200,
});

// ── 1. Task: call the model ────────────────────────────────────────
const callModel = task("callModel", async (messages: BaseMessageLike[]) => {
  const response = await model.invoke(messages);
  return response;
});

// ── 2. Chatbot with memory (functional style) ─────────────────────
const checkpointer = new MemorySaver();

const chatbot = entrypoint({
  name: "chatbot",
  checkpointer,   // Enable persistence
}, async (inputs: BaseMessageLike[]) => {
  // getPreviousState() returns what was saved in the LAST invocation
  // First invocation: undefined (no previous state)
  const previous = getPreviousState<BaseMessage[]>() ?? [];

  // Merge previous messages with new input
  const messages = addMessages(previous, inputs);

  // Call the model with full conversation history
  const response = await callModel(messages);

  // entrypoint.final() separates:
  //   value → what invoke() returns to the caller
  //   save  → what gets persisted for the NEXT invocation
  return entrypoint.final({
    value: response,                       // Caller gets just the AI response
    save: addMessages(messages, response), // Full history saved for next time
  });
});

// ── 3. Test: multi-turn conversation ───────────────────────────────
console.log("=== Functional API: Memory ===\n");

const config = { configurable: { thread_id: "chat-1" } };

// Turn 1
console.log("--- Turn 1 ---");
let response = await chatbot.invoke(
  [{ role: "user", content: "Hi! I'm Mansi, I'm learning LangGraph." }],
  config
);
console.log("AI:", response.content);

// Turn 2 — chatbot remembers!
console.log("\n--- Turn 2 ---");
response = await chatbot.invoke(
  [{ role: "user", content: "What's my name and what am I learning?" }],
  config
);
console.log("AI:", response.content);

// Turn 3 — context builds up
console.log("\n--- Turn 3 ---");
response = await chatbot.invoke(
  [{ role: "user", content: "Recommend one thing I should learn next." }],
  config
);
console.log("AI:", response.content);

// Different thread — no shared memory
console.log("\n--- Different Thread ---");
response = await chatbot.invoke(
  [{ role: "user", content: "Do you know my name?" }],
  { configurable: { thread_id: "chat-2" } }
);
console.log("AI:", response.content);
