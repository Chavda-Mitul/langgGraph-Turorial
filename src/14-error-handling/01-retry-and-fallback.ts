/**
 * 14-error-handling/01-retry-and-fallback.ts
 * ───────────────────────────────────────────
 * Error Handling: Retry logic and fallback strategies.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THEORY: Why Error Handling Matters in AI Graphs
 * ═══════════════════════════════════════════════════════════════════
 *
 * AI applications fail in UNIQUE ways compared to traditional software:
 *
 * 1. LLM API FAILURES
 *    - Rate limits (429 errors)
 *    - Timeouts (model overloaded)
 *    - Network errors
 *    → Solution: RETRY with exponential backoff
 *
 * 2. LLM OUTPUT FAILURES
 *    - Invalid JSON output
 *    - Hallucinated tool names
 *    - Unexpected format
 *    → Solution: VALIDATION + retry with corrective prompt
 *
 * 3. TOOL EXECUTION FAILURES
 *    - External API down
 *    - Invalid arguments from LLM
 *    - Permission errors
 *    → Solution: FALLBACK nodes (graceful degradation)
 *
 * 4. INFINITE LOOPS
 *    - Agent keeps calling tools without progress
 *    - Revision loop never converges
 *    → Solution: MAX ITERATIONS + circuit breaker
 *
 * Error Handling Strategies in LangGraph:
 *
 *   ┌─────────────────────────────────────────┐
 *   │  Strategy        │  When to Use          │
 *   ├─────────────────────────────────────────┤
 *   │  Try/Catch       │  Known failure modes   │
 *   │  Retry (N times) │  Transient failures    │
 *   │  Fallback Node   │  Alternative approach  │
 *   │  State Tracking  │  Resume after failure  │
 *   │  Max Iterations  │  Prevent infinite loops│
 *   └─────────────────────────────────────────┘
 *
 * ═══════════════════════════════════════════════════════════════════
 *
 * Run: npx ts-node --esm src/14-error-handling/01-retry-and-fallback.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  Annotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  maxTokens: 200,
});

// ══════════════════════════════════════════════════════════════════
// PATTERN 1: Retry with Validation
// ══════════════════════════════════════════════════════════════════
// Ask the LLM for structured output, retry if it gives bad format.

const JsonState = Annotation.Root({
  prompt: Annotation<string>(),
  rawOutput: Annotation<string>(),
  parsedData: Annotation<Record<string, unknown> | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),
  attempts: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),
  error: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),
});

async function generateJson(state: typeof JsonState.State) {
  const attempt = state.attempts + 1;
  console.log(`🔄 Attempt ${attempt}: Generating JSON...`);

  const correctionHint = state.error
    ? `\n\nYour previous response was invalid: ${state.error}\nPlease fix it and return ONLY valid JSON.`
    : "";

  const res = await model.invoke(
    `${state.prompt}${correctionHint}\n\nRespond with ONLY a valid JSON object, no markdown.`
  );

  return {
    rawOutput: res.content as string,
    attempts: attempt,
  };
}

function validateJson(state: typeof JsonState.State): string {
  try {
    // Try to extract JSON from the response
    const jsonMatch = state.rawOutput.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error("No JSON object found in response");
    }
    JSON.parse(jsonMatch[0]);
    return "success";
  } catch (e) {
    if (state.attempts >= 3) {
      console.log("❌ Max retries reached, using fallback");
      return "fallback";
    }
    console.log(`⚠️  Invalid JSON, retrying... (${e instanceof Error ? e.message : e})`);
    return "retry";
  }
}

function parseSuccess(state: typeof JsonState.State) {
  console.log("✅ JSON parsed successfully!");
  const jsonMatch = state.rawOutput.match(/\{[\s\S]*\}/)!;
  return {
    parsedData: JSON.parse(jsonMatch[0]),
    error: "",
  };
}

function parseFallback(state: typeof JsonState.State) {
  console.log("🔄 Using fallback: wrapping raw output as JSON");
  return {
    parsedData: { raw: state.rawOutput, fallback: true },
    error: "Failed to parse after max retries, used fallback",
  };
}

function setRetryError(state: typeof JsonState.State) {
  return {
    error: `Could not parse as JSON: "${state.rawOutput.slice(0, 100)}..."`,
  };
}

const jsonGraph = new StateGraph(JsonState)
  .addNode("generate", generateJson)
  .addNode("parse_success", parseSuccess)
  .addNode("parse_fallback", parseFallback)
  .addNode("set_retry_error", setRetryError)
  .addEdge(START, "generate")
  .addConditionalEdges("generate", validateJson, [
    "parse_success",
    "parse_fallback",
    "set_retry_error",
  ])
  // Map the conditional edge names to actual targets
  .addEdge("parse_success", END)
  .addEdge("parse_fallback", END)
  .addEdge("set_retry_error", "generate")  // Retry loop!
  .compile();

// ══════════════════════════════════════════════════════════════════
// PATTERN 2: Try/Catch in Nodes with Graceful Degradation
// ══════════════════════════════════════════════════════════════════

const ServiceState = Annotation.Root({
  query: Annotation<string>(),
  primaryResult: Annotation<string>(),
  fallbackUsed: Annotation<boolean>({
    reducer: (_, b) => b,
    default: () => false,
  }),
  status: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "pending",
  }),
});

// Simulates a flaky external service
async function callPrimaryService(state: typeof ServiceState.State) {
  console.log("🌐 Calling primary service...");

  try {
    // Simulate: 50% chance of failure
    if (Math.random() < 0.5) {
      throw new Error("Service temporarily unavailable (simulated)");
    }

    const res = await model.invoke(
      `Answer this concisely: ${state.query}`
    );
    return {
      primaryResult: res.content as string,
      status: "success",
    };
  } catch (error) {
    // Don't crash the graph! Catch and route to fallback.
    console.log(`⚠️  Primary service failed: ${error instanceof Error ? error.message : error}`);
    return {
      primaryResult: "",
      status: "primary_failed",
    };
  }
}

function routeAfterPrimary(state: typeof ServiceState.State): string {
  if (state.status === "success") return "done";
  return "fallback";
}

async function callFallbackService(state: typeof ServiceState.State) {
  console.log("🔄 Using fallback service...");
  const res = await model.invoke(
    `Provide a brief answer: ${state.query}`
  );
  return {
    primaryResult: res.content as string,
    fallbackUsed: true,
    status: "fallback_success",
  };
}

function done(_state: typeof ServiceState.State) {
  return {}; // passthrough
}

const serviceGraph = new StateGraph(ServiceState)
  .addNode("primary", callPrimaryService)
  .addNode("fallback", callFallbackService)
  .addNode("done", done)
  .addEdge(START, "primary")
  .addConditionalEdges("primary", routeAfterPrimary, ["done", "fallback"])
  .addEdge("fallback", "done")
  .addEdge("done", END)
  .compile();

// ══════════════════════════════════════════════════════════════════
// Run both patterns
// ══════════════════════════════════════════════════════════════════

console.log("=== Error Handling Patterns ===\n");

// Pattern 1: Retry with validation
console.log("─── Pattern 1: Retry with JSON Validation ───\n");
const jsonResult = await jsonGraph.invoke({
  prompt: 'Generate a JSON object with fields: name (string), age (number), skills (array of strings). Use the name "Mansi".',
});
console.log("\nResult:");
console.log("  Parsed data:", JSON.stringify(jsonResult.parsedData, null, 2));
console.log("  Attempts:", jsonResult.attempts);
console.log("  Error:", jsonResult.error || "None");

// Pattern 2: Try/catch with fallback
console.log("\n─── Pattern 2: Service with Fallback ───\n");
const serviceResult = await serviceGraph.invoke({
  query: "What is the capital of France?",
});
console.log("\nResult:");
console.log("  Answer:", serviceResult.primaryResult);
console.log("  Fallback used:", serviceResult.fallbackUsed);
console.log("  Status:", serviceResult.status);
