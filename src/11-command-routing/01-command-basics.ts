/**
 * 11-command-routing/01-command-basics.ts
 * ────────────────────────────────────────
 * Command: The modern way to route between nodes.
 *
 * Key Concepts:
 * - Command({ goto, update }): Route to a node AND update state in one shot
 * - Replaces addConditionalEdges() for many use cases
 * - goto: which node to visit next (or END to finish)
 * - update: state changes to apply
 * - addNode("name", fn, { ends: [...] }): Declare possible destinations
 *   (required for visualization and Command-based routing)
 *
 * Old way: addConditionalEdges("node", routerFn, [...destinations])
 * New way: return new Command({ goto: "destination", update: {...} })
 *
 * Run: npx ts-node --esm src/11-command-routing/01-command-basics.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  Annotation,
  StateGraph,
  MessagesAnnotation,
  Command,
  START,
  END,
} from "@langchain/langgraph";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.3,
  maxTokens: 200,
});

// ── 1. State ───────────────────────────────────────────────────────
const RouterState = Annotation.Root({
  ...MessagesAnnotation.spec,
  category: Annotation<string>(),
  result: Annotation<string>(),
});

// ── 2. Triage node — uses Command to route ─────────────────────────
async function triage(state: typeof RouterState.State) {
  const lastMsg = state.messages[state.messages.length - 1];

  const res = await model.invoke(
    `Classify this request as "code", "math", or "chat": "${lastMsg.content}". Reply with one word.`
  );

  const category = (res.content as string).toLowerCase().trim();
  console.log(`🔀 Triage: Routing to "${category}"`);

  // Command: choose destination AND update state in one return
  return new Command({
    goto: category.includes("code") ? "codeExpert" :
          category.includes("math") ? "mathExpert" :
          "chatBot",
    update: {
      category,
    },
  });
}

// ── 3. Specialist nodes — each returns Command to end ──────────────
async function codeExpert(state: typeof RouterState.State) {
  const lastMsg = state.messages[state.messages.length - 1];
  const res = await model.invoke(`As a coding expert, answer: ${lastMsg.content}`);

  return new Command({
    goto: END,
    update: {
      result: res.content as string,
      messages: [new AIMessage(`[Code Expert] ${res.content}`)],
    },
  });
}

async function mathExpert(state: typeof RouterState.State) {
  const lastMsg = state.messages[state.messages.length - 1];
  const res = await model.invoke(`As a math expert, answer: ${lastMsg.content}`);

  return new Command({
    goto: END,
    update: {
      result: res.content as string,
      messages: [new AIMessage(`[Math Expert] ${res.content}`)],
    },
  });
}

async function chatBot(state: typeof RouterState.State) {
  const lastMsg = state.messages[state.messages.length - 1];
  const res = await model.invoke(`Chat casually: ${lastMsg.content}`);

  return new Command({
    goto: END,
    update: {
      result: res.content as string,
      messages: [new AIMessage(`[Chat] ${res.content}`)],
    },
  });
}

// ── 4. Build graph with `ends` declarations ────────────────────────
// When using Command, you declare possible destinations with `ends`
const graph = new StateGraph(RouterState)
  .addNode("triage", triage, {
    ends: ["codeExpert", "mathExpert", "chatBot"], // Possible Command targets
  })
  .addNode("codeExpert", codeExpert, { ends: [END] })
  .addNode("mathExpert", mathExpert, { ends: [END] })
  .addNode("chatBot", chatBot, { ends: [END] })
  .addEdge(START, "triage")
  .compile();

// ── 5. Test ────────────────────────────────────────────────────────
console.log("=== Command Routing ===\n");

const queries = [
  "Write a TypeScript function to reverse a string",
  "What is the integral of x^2?",
  "Hey, how's your day going?",
];

for (const q of queries) {
  console.log(`\n📨 Query: "${q}"`);
  const result = await graph.invoke({
    messages: [new HumanMessage(q)],
  });
  console.log(`📂 Category: ${result.category}`);
  console.log(`💬 Result: ${result.result?.slice(0, 150)}...`);
  console.log("─".repeat(60));
}
