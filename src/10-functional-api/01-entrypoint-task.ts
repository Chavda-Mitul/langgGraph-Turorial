/**
 * 10-functional-api/01-entrypoint-task.ts
 * ─────────────────────────────────────────
 * Functional API: Build workflows with `entrypoint` + `task`.
 *
 * Key Concepts:
 * - entrypoint(): Defines a workflow — like main() for your graph
 * - task(): Defines a unit of work — auto-tracked, auto-streamed
 * - No StateGraph, no edges — just normal async/await code!
 * - Tasks are automatically visible in streaming & tracing
 * - entrypoint is the NEW recommended way for simpler workflows
 *
 * StateGraph API = declarative (define graph structure)
 * Functional API = imperative (write regular code)
 *
 * Run: npx ts-node --esm src/10-functional-api/01-entrypoint-task.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { task, entrypoint } from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 200,
});

// ── 1. Define tasks ────────────────────────────────────────────────
// task() wraps a function — gives it a name for streaming/tracing.
// It's like a node in StateGraph, but without the graph boilerplate.

const generateJoke = task("generateJoke", async (topic: string) => {
  console.log("  🎭 Generating joke...");
  const msg = await model.invoke(`Write a short, funny joke about ${topic}`);
  return msg.content as string;
});

const critiqueJoke = task("critiqueJoke", async (joke: string) => {
  console.log("  🧐 Critiquing joke...");
  const msg = await model.invoke(
    `Rate this joke from 1-10 and suggest one improvement:\n\n${joke}`
  );
  return msg.content as string;
});

const improveJoke = task("improveJoke", async (args: { joke: string; critique: string }) => {
  console.log("  ✨ Improving joke...");
  const msg = await model.invoke(
    `Improve this joke based on the feedback:\n\nJoke: ${args.joke}\n\nFeedback: ${args.critique}\n\nWrite only the improved joke.`
  );
  return msg.content as string;
});

// ── 2. Define the workflow with entrypoint ─────────────────────────
// entrypoint() is the "main" of your workflow.
// Inside it, you call tasks like regular functions.

const jokeWorkflow = entrypoint(
  "jokeImprover",
  async (topic: string) => {
    // Step 1: Generate initial joke
    const originalJoke = await generateJoke(topic);

    // Step 2: Get critique
    const critique = await critiqueJoke(originalJoke);

    // Step 3: Improve based on feedback
    const improvedJoke = await improveJoke({ joke: originalJoke, critique });

    // Return final result
    return {
      original: originalJoke,
      critique,
      improved: improvedJoke,
    };
  }
);

// ── 3. Run the workflow ────────────────────────────────────────────
console.log("=== Functional API: entrypoint + task ===\n");

// invoke() works just like StateGraph
const result = await jokeWorkflow.invoke("programmers");

console.log("\n📝 Original Joke:\n", result.original);
console.log("\n🧐 Critique:\n", result.critique);
console.log("\n✨ Improved Joke:\n", result.improved);

// ── 4. Streaming also works! ───────────────────────────────────────
console.log("\n\n=== Streaming with Functional API ===\n");

const stream = await jokeWorkflow.stream("TypeScript", {
  streamMode: "updates",
});

for await (const step of stream) {
  const [taskName] = Object.entries(step);
  console.log(`📍 Task completed: ${taskName}`);
}
