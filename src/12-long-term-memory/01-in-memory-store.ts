/**
 * 12-long-term-memory/01-in-memory-store.ts
 * ───────────────────────────────────────────
 * InMemoryStore: Cross-thread persistent memory.
 *
 * Key Concepts:
 * - InMemoryStore: Key-value store that persists ACROSS threads
 * - Namespaces: Organize memory like folders — ["userId", "preferences"]
 * - store.put(namespace, key, value): Save data
 * - store.get(namespace, key): Retrieve data
 * - store.search(namespace): List all items in a namespace
 * - Access store in nodes via config.store
 *
 * MemorySaver = short-term (within a thread/conversation)
 * InMemoryStore = long-term (across all threads/conversations)
 *
 * Run: npx ts-node --esm src/12-long-term-memory/01-in-memory-store.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  MemorySaver,
  InMemoryStore,
} from "@langchain/langgraph";
import type { LangGraphRunnableConfig } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 200,
});

// ── 1. Create stores ──────────────────────────────────────────────
const checkpointer = new MemorySaver();  // Short-term (per thread)
const store = new InMemoryStore();        // Long-term (cross-thread)

// ── 2. Graph nodes that use the store ──────────────────────────────
async function chatNode(
  state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) {
  // Access the store from config
  const memoryStore = config.store!;
  const userId = config.configurable?.user_id || "anonymous";
  const namespace = [userId, "preferences"];

  // Load user preferences from long-term memory
  const items = await memoryStore.search(namespace);
  const preferences = items.map(item => JSON.stringify(item.value)).join(", ");

  const systemMsg = preferences
    ? `You are a helpful assistant. User preferences: ${preferences}. Tailor your responses accordingly.`
    : "You are a helpful assistant. You don't know much about this user yet.";

  const response = await model.invoke([
    { role: "system", content: systemMsg },
    ...state.messages,
  ]);

  return { messages: [response] };
}

async function learnPreferences(
  state: typeof MessagesAnnotation.State,
  config: LangGraphRunnableConfig
) {
  const memoryStore = config.store!;
  const userId = config.configurable?.user_id || "anonymous";
  const namespace = [userId, "preferences"];

  // Use LLM to extract preferences from conversation
  const lastMsg = state.messages[state.messages.length - 2]; // User's message
  const extraction = await model.invoke(
    `Extract any user preferences from this message. If there are preferences, return them as JSON like {"preference": "value"}. If no preferences, return "none".\n\nMessage: "${lastMsg.content}"`
  );

  const content = extraction.content as string;
  if (!content.toLowerCase().includes("none") && content.includes("{")) {
    try {
      const prefs = JSON.parse(content.match(/\{[^}]+\}/)?.[0] || "{}");
      for (const [key, value] of Object.entries(prefs)) {
        await memoryStore.put(namespace, key, { [key]: value });
        console.log(`  💾 Saved preference: ${key} = ${value}`);
      }
    } catch {
      // Not valid JSON, skip
    }
  }

  return {}; // No state changes, just store updates
}

// ── 3. Build graph ─────────────────────────────────────────────────
const graph = new StateGraph(MessagesAnnotation)
  .addNode("chat", chatNode)
  .addNode("learn", learnPreferences)
  .addEdge(START, "chat")
  .addEdge("chat", "learn")
  .addEdge("learn", END)
  .compile({
    checkpointer,     // Thread-level memory
    store,            // Cross-thread memory
  });

// ── 4. Test: cross-thread memory ───────────────────────────────────
console.log("=== Long-Term Memory: InMemoryStore ===\n");

const userId = "mansi";

// Thread 1: User shares preferences
console.log("--- Thread 1 (sharing preferences) ---");
let result = await graph.invoke(
  { messages: [new HumanMessage("I prefer concise answers and I love TypeScript examples.")] },
  { configurable: { thread_id: "thread-1", user_id: userId } }
);
console.log("AI:", result.messages[result.messages.length - 1].content);

// Thread 2: NEW conversation, but store remembers preferences!
console.log("\n--- Thread 2 (new conversation, same user) ---");
result = await graph.invoke(
  { messages: [new HumanMessage("Explain what a Promise is in JavaScript.")] },
  { configurable: { thread_id: "thread-2", user_id: userId } }
);
console.log("AI:", result.messages[result.messages.length - 1].content);

// Thread 3: Different user — no shared preferences
console.log("\n--- Thread 3 (different user) ---");
result = await graph.invoke(
  { messages: [new HumanMessage("Explain what a Promise is.")] },
  { configurable: { thread_id: "thread-3", user_id: "rahul" } }
);
console.log("AI:", result.messages[result.messages.length - 1].content);

// Inspect store contents
console.log("\n--- Store Contents ---");
const mansiPrefs = await store.search([userId, "preferences"]);
console.log(`Mansi's preferences (${mansiPrefs.length} items):`);
mansiPrefs.forEach(item => console.log(`  ${item.key}:`, item.value));
