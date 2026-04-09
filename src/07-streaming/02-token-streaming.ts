/**
 * 07-streaming/02-token-streaming.ts
 * ────────────────────────────────────
 * Token Streaming: Stream LLM tokens in real-time.
 *
 * Key Concepts:
 * - graph.streamEvents(): Fine-grained event streaming
 * - Events include: on_llm_start, on_llm_stream, on_llm_end
 * - on_chat_model_stream: Get individual tokens as they generate
 * - Great for building ChatGPT-like real-time UIs
 *
 * Run: npx ts-node --esm src/07-streaming/02-token-streaming.ts
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

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 200,
});

// ── 1. Simple chat graph ───────────────────────────────────────────
const graph = new StateGraph(MessagesAnnotation)
  .addNode("chat", async (state) => {
    const response = await model.invoke(state.messages);
    return { messages: [response] };
  })
  .addEdge(START, "chat")
  .addEdge("chat", END)
  .compile();

// ── 2. Stream events (token-level) ────────────────────────────────
console.log("=== Token Streaming ===\n");
console.log("Streaming response:");
process.stdout.write("🤖 ");

const eventStream = graph.streamEvents(
  { messages: [new HumanMessage("Write a haiku about AI agents building graphs.")] },
  { version: "v2" }
);

for await (const event of eventStream) {
  // on_chat_model_stream fires for each token
  if (event.event === "on_chat_model_stream") {
    const token = event.data.chunk?.content;
    if (token) {
      process.stdout.write(token);
    }
  }
}

console.log("\n\n--- Event Types ---");

// ── 3. Show all event types ────────────────────────────────────────
// Run again to display event types (for learning)
const events = graph.streamEvents(
  { messages: [new HumanMessage("Say hello in 5 words.")] },
  { version: "v2" }
);

const eventTypes = new Set<string>();
for await (const event of events) {
  eventTypes.add(event.event);
}

console.log("Observed events:");
eventTypes.forEach(e => console.log(`  - ${e}`));
