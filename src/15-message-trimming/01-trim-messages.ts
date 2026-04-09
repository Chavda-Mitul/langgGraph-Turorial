/**
 * 15-message-trimming/01-trim-messages.ts
 * ─────────────────────────────────────────
 * Message Trimming: Manage context window for long conversations.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THEORY: The Context Window Problem
 * ═══════════════════════════════════════════════════════════════════
 *
 * Every LLM has a CONTEXT WINDOW — a maximum number of tokens it can
 * process at once. For example:
 *   - GPT-4o: 128K tokens
 *   - Claude 3.5: 200K tokens
 *   - Llama 3: 8K-128K tokens (varies)
 *
 * In a chatbot with memory (MemorySaver), messages ACCUMULATE:
 *   Turn 1: 1 user + 1 AI = ~200 tokens
 *   Turn 10: 10 user + 10 AI = ~2,000 tokens
 *   Turn 100: 100 user + 100 AI = ~20,000 tokens
 *   ...eventually BOOM! Context window exceeded.
 *
 * Solutions:
 *
 * ┌──────────────────────┬────────────────────────────────────┐
 * │ Strategy             │ How it works                       │
 * ├──────────────────────┼────────────────────────────────────┤
 * │ trimMessages()       │ Keep only the last N messages      │
 * │                      │ or last N tokens                   │
 * ├──────────────────────┼────────────────────────────────────┤
 * │ Summarization        │ Summarize old messages, keep       │
 * │                      │ summary + recent messages          │
 * ├──────────────────────┼────────────────────────────────────┤
 * │ Sliding Window       │ Always keep a fixed window of      │
 * │                      │ the most recent messages            │
 * ├──────────────────────┼────────────────────────────────────┤
 * │ Token Counting       │ Remove oldest messages until       │
 * │                      │ total tokens fit the budget        │
 * └──────────────────────┴────────────────────────────────────┘
 *
 * IMPORTANT: Always keep the SystemMessage (if any) — it contains
 * the agent's personality and instructions!
 *
 * ═══════════════════════════════════════════════════════════════════
 *
 * Key Concepts:
 * - trimMessages(): Built-in utility from @langchain/core/messages
 * - maxTokens: Maximum token count to keep
 * - strategy: "last" (keep newest) — the most common strategy
 * - includeSystem: Always keep the system message (true by default)
 * - Custom trimming: Build your own with simple array slicing
 *
 * Run: npx ts-node --esm src/15-message-trimming/01-trim-messages.ts
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
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  trimMessages,
} from "@langchain/core/messages";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 200,
});

// ══════════════════════════════════════════════════════════════════
// PATTERN 1: Simple Message Count Trimming
// ══════════════════════════════════════════════════════════════════
// Keep only the last N messages. Simple but effective.

const simpleTrimGraph = new StateGraph(MessagesAnnotation)
  .addNode("chat", async (state) => {
    // Trim to last 6 messages (3 turns of user+AI)
    // This runs BEFORE sending to the LLM, not stored in state
    const trimmedMessages = state.messages.slice(-6);

    console.log(`  📊 Messages: ${state.messages.length} total, sending ${trimmedMessages.length} to LLM`);

    const response = await model.invoke([
      { role: "system", content: "You are a helpful assistant. Be concise." },
      ...trimmedMessages,
    ]);

    return { messages: [response] };
  })
  .addEdge(START, "chat")
  .addEdge("chat", END)
  .compile({
    checkpointer: new MemorySaver(),
  });

// ══════════════════════════════════════════════════════════════════
// PATTERN 2: Using trimMessages() utility
// ══════════════════════════════════════════════════════════════════
// The built-in trimMessages offers more control.

const smartTrimGraph = new StateGraph(MessagesAnnotation)
  .addNode("chat", async (state) => {
    // trimMessages: keep newest messages that fit in token budget
    const trimmed = await trimMessages(state.messages, {
      // Maximum number of tokens to keep
      maxTokens: 500,
      // Strategy: "last" keeps the most recent messages
      strategy: "last",
      // Token counter — estimate tokens (4 chars ≈ 1 token)
      tokenCounter: (msgs) => {
        return msgs.reduce((acc, msg) => {
          const content = typeof msg.content === "string"
            ? msg.content
            : JSON.stringify(msg.content);
          return acc + Math.ceil(content.length / 4);
        }, 0);
      },
      // Always keep the system message at the top
      includeSystem: true,
      // Keep at least the last message (the user's latest question)
      allowPartial: false,
    });

    console.log(`  📊 Trimmed: ${state.messages.length} → ${trimmed.length} messages`);

    const response = await model.invoke([
      { role: "system", content: "You are a helpful assistant. Remember context from the conversation." },
      ...trimmed,
    ]);

    return { messages: [response] };
  })
  .addEdge(START, "chat")
  .addEdge("chat", END)
  .compile({
    checkpointer: new MemorySaver(),
  });

// ══════════════════════════════════════════════════════════════════
// PATTERN 3: Summarize + Trim (Advanced)
// ══════════════════════════════════════════════════════════════════
// Summarize old messages, then keep summary + recent messages.
// Best of both worlds: context preserved, tokens saved.

import { Annotation } from "@langchain/langgraph";

const SummaryState = Annotation.Root({
  ...MessagesAnnotation.spec,
  summary: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),
});

const summaryGraph = new StateGraph(SummaryState)
  .addNode("chat", async (state) => {
    // Build prompt with summary context
    const systemContent = state.summary
      ? `You are a helpful assistant. Here's a summary of the conversation so far:\n\n${state.summary}\n\nUse this context to inform your responses.`
      : "You are a helpful assistant.";

    // Only send the last 4 messages to the LLM (+ system with summary)
    const recentMessages = state.messages.slice(-4);

    console.log(`  📊 Using summary + ${recentMessages.length} recent messages`);

    const response = await model.invoke([
      { role: "system", content: systemContent },
      ...recentMessages,
    ]);

    return { messages: [response] };
  })
  .addNode("maybe_summarize", async (state) => {
    // Summarize if we have more than 8 messages
    if (state.messages.length <= 8) {
      return {};
    }

    console.log("  📋 Summarizing old messages...");

    // Get old messages (everything except the last 4)
    const oldMessages = state.messages.slice(0, -4);
    const oldContext = oldMessages
      .map(m => `${m.getType()}: ${m.content}`)
      .join("\n");

    const previousSummary = state.summary
      ? `Previous summary: ${state.summary}\n\n`
      : "";

    const res = await model.invoke(
      `${previousSummary}Summarize this conversation in 2-3 sentences, capturing key facts:\n\n${oldContext}`
    );

    return { summary: res.content as string };
  })
  .addEdge(START, "chat")
  .addEdge("chat", "maybe_summarize")
  .addEdge("maybe_summarize", END)
  .compile({
    checkpointer: new MemorySaver(),
  });

// ══════════════════════════════════════════════════════════════════
// Run demonstrations
// ══════════════════════════════════════════════════════════════════

console.log("=== Message Trimming & Context Management ===\n");

// Demo 1: Simple trimming
console.log("─── Pattern 1: Simple Count-Based Trimming ───\n");
const config1 = { configurable: { thread_id: "trim-1" } };

const questions = [
  "Hi! My name is Mansi.",
  "I'm learning LangGraph in TypeScript.",
  "I love building AI agents.",
  "What have I told you about myself?",
];

for (const q of questions) {
  console.log(`\n👤 ${q}`);
  const res = await simpleTrimGraph.invoke(
    { messages: [new HumanMessage(q)] },
    config1
  );
  console.log(`🤖 ${res.messages[res.messages.length - 1].content}`);
}

// Demo 2: Summary-based trimming
console.log("\n\n─── Pattern 3: Summarize + Trim ───\n");
const config3 = { configurable: { thread_id: "summary-1" } };

const longConvo = [
  "Hi! I'm Mansi and I'm a developer from India.",
  "I work with TypeScript and Python mostly.",
  "My current project is a LangGraph tutorial.",
  "I want to teach people about AI agents.",
  "What do you know about me so far?",
];

for (const q of longConvo) {
  console.log(`\n👤 ${q}`);
  const res = await summaryGraph.invoke(
    { messages: [new HumanMessage(q)] },
    config3
  );
  console.log(`🤖 ${res.messages[res.messages.length - 1].content}`);
}

// Check if summary was generated
const summaryState = await summaryGraph.getState(config3);
if (summaryState.values.summary) {
  console.log("\n📋 Stored Summary:", summaryState.values.summary);
}
