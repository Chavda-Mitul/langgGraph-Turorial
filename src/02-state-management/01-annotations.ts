/**
 * 02-state-management/01-annotations.ts
 * ──────────────────────────────────────
 * Annotations: Define your graph's state schema.
 *
 * Key Concepts:
 * - Annotation.Root(): Creates a state schema (like an interface, but runtime)
 * - Annotation<T>(): Defines a single field with type T
 * - Fields without reducers: new values OVERWRITE old values
 * - Fields with reducers: new values MERGE with old values
 * - MessagesAnnotation: A prebuilt annotation for chat message lists
 *
 * Run: npx ts-node --esm src/02-state-management/01-annotations.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  Annotation,
  StateGraph,
  MessagesAnnotation,
  START,
  END,
} from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.5,
  maxTokens: 200,
});

// ── 1. Simple Annotation (overwrite behavior) ──────────────────────
// Fields WITHOUT reducers: last write wins
const SimpleState = Annotation.Root({
  name: Annotation<string>(),      // Overwritten each time
  age: Annotation<number>(),       // Overwritten each time
  greeting: Annotation<string>(),  // Overwritten each time
});

const simpleGraph = new StateGraph(SimpleState)
  .addNode("greeter", async (state) => {
    console.log(`  Greeter sees: name=${state.name}, age=${state.age}`);
    return { greeting: `Hello ${state.name}, you are ${state.age} years old!` };
  })
  .addEdge(START, "greeter")
  .addEdge("greeter", END)
  .compile();

const simpleResult = await simpleGraph.invoke({ name: "Mansi", age: 25 });
console.log("=== Simple Annotation ===");
console.log("Result:", simpleResult.greeting);

// ── 2. MessagesAnnotation (built-in, with reducer) ────────────────
// MessagesAnnotation has a `messages` field with a concat reducer.
// Every message you return gets APPENDED, not replaced.
const chatGraph = new StateGraph(MessagesAnnotation)
  .addNode("chat", async (state) => {
    // state.messages contains ALL messages so far
    console.log(`\n  Chat node sees ${state.messages.length} message(s)`);
    const response = await model.invoke(state.messages);
    return { messages: [response] }; // Appended!
  })
  .addEdge(START, "chat")
  .addEdge("chat", END)
  .compile();

const chatResult = await chatGraph.invoke({
  messages: [new HumanMessage("What is TypeScript in one line?")],
});

console.log("\n=== MessagesAnnotation ===");
console.log("Total messages:", chatResult.messages.length); // 2: human + AI
console.log("AI says:", chatResult.messages[1].content);

// ── 3. Extending MessagesAnnotation ────────────────────────────────
// You can ADD extra fields alongside the built-in messages field.
const ExtendedState = Annotation.Root({
  ...MessagesAnnotation.spec,   // Inherit messages with its reducer
  username: Annotation<string>(),
  mood: Annotation<string>(),
});

const extendedGraph = new StateGraph(ExtendedState)
  .addNode("detect_mood", async (state) => {
    const lastMsg = state.messages[state.messages.length - 1];
    const response = await model.invoke(
      `In one word, what is the mood of this message? "${lastMsg.content}"`
    );
    return { mood: response.content as string };
  })
  .addNode("respond", async (state) => {
    const response = await model.invoke([
      { role: "system", content: `User's name: ${state.username}. Their mood: ${state.mood}. Respond warmly in 1-2 sentences.` },
      ...state.messages,
    ]);
    return { messages: [response] };
  })
  .addEdge(START, "detect_mood")
  .addEdge("detect_mood", "respond")
  .addEdge("respond", END)
  .compile();

const extResult = await extendedGraph.invoke({
  messages: [new HumanMessage("I just finished my first TypeScript project!")],
  username: "Mansi",
});

console.log("\n=== Extended MessagesAnnotation ===");
console.log("Detected mood:", extResult.mood);
console.log("Response:", extResult.messages[extResult.messages.length - 1].content);
