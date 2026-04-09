/**
 * 10-functional-api/03-branching-functional.ts
 * ──────────────────────────────────────────────
 * Functional API Patterns: Branching, Parallelism, and Loops.
 *
 * Key Concepts:
 * - Branching: Just use if/else — no conditional edges needed!
 * - Parallelism: Use Promise.all() — tasks run concurrently
 * - Loops: Use while/for — no need to wire edges back
 * - The functional API makes these patterns feel NATURAL
 *
 * Run: npx ts-node --esm src/10-functional-api/03-branching-functional.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { task, entrypoint } from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 200,
});

// ── Tasks ──────────────────────────────────────────────────────────
const classify = task("classify", async (text: string) => {
  const res = await model.invoke(
    `Classify this as "technical" or "creative": "${text}". Reply with one word only.`
  );
  return (res.content as string).toLowerCase().trim();
});

const technicalResponse = task("technicalResponse", async (text: string) => {
  const res = await model.invoke(`Give a precise, factual answer to: ${text}`);
  return res.content as string;
});

const creativeResponse = task("creativeResponse", async (text: string) => {
  const res = await model.invoke(`Give a creative, imaginative response to: ${text}`);
  return res.content as string;
});

const summarize = task("summarize", async (text: string) => {
  const res = await model.invoke(`Summarize in one sentence: ${text}`);
  return res.content as string;
});

const analyze = task("analyze", async (text: string) => {
  const res = await model.invoke(`What is the key insight in: ${text}`);
  return res.content as string;
});

const refine = task("refine", async (args: { draft: string; feedback: string }) => {
  const res = await model.invoke(
    `Improve this draft: ${args.draft}\n\nFeedback: ${args.feedback}`
  );
  return res.content as string;
});

const review = task("review", async (text: string) => {
  const res = await model.invoke(
    `Review this briefly. If good say "APPROVED", else give one improvement: ${text}`
  );
  return res.content as string;
});

// ── 1. Branching: Just if/else! ────────────────────────────────────
const branchingWorkflow = entrypoint(
  "smartRouter",
  async (input: string) => {
    const category = await classify(input);
    console.log(`  Classified as: ${category}`);

    // Plain JavaScript branching — no addConditionalEdges needed
    if (category.includes("technical")) {
      return { category, response: await technicalResponse(input) };
    } else {
      return { category, response: await creativeResponse(input) };
    }
  }
);

// ── 2. Parallelism: Just Promise.all! ──────────────────────────────
const parallelWorkflow = entrypoint(
  "parallelAnalysis",
  async (text: string) => {
    // Both tasks run at the SAME TIME
    const [summary, insight] = await Promise.all([
      summarize(text),
      analyze(text),
    ]);

    return { summary, insight };
  }
);

// ── 3. Loops: Just while loops! ────────────────────────────────────
const loopWorkflow = entrypoint(
  "refineUntilApproved",
  async (topic: string) => {
    let draft = await creativeResponse(topic);
    let attempts = 0;
    const maxAttempts = 3;

    while (attempts < maxAttempts) {
      attempts++;
      const feedback = await review(draft);

      if (feedback.toUpperCase().includes("APPROVED")) {
        return { draft, attempts, approved: true };
      }

      draft = await refine({ draft, feedback });
    }

    return { draft, attempts, approved: false };
  }
);

// ── Run all patterns ───────────────────────────────────────────────
console.log("=== Functional API Patterns ===\n");

// Branching
console.log("--- 1. Branching ---");
const branchResult = await branchingWorkflow.invoke("How does garbage collection work in Node.js?");
console.log(`Category: ${branchResult.category}`);
console.log(`Response: ${branchResult.response}\n`);

// Parallel
console.log("--- 2. Parallelism ---");
const parallelResult = await parallelWorkflow.invoke(
  "AI agents use graphs for orchestration, enabling complex multi-step reasoning with tool use."
);
console.log(`Summary: ${parallelResult.summary}`);
console.log(`Insight: ${parallelResult.insight}\n`);

// Loop
console.log("--- 3. Loop (refine until approved) ---");
const loopResult = await loopWorkflow.invoke("Write a tagline for LangGraph");
console.log(`Final: ${loopResult.draft}`);
console.log(`Attempts: ${loopResult.attempts}, Approved: ${loopResult.approved}`);
