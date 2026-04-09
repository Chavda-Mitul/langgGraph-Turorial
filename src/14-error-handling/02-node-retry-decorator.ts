/**
 * 14-error-handling/02-node-retry-decorator.ts
 * ──────────────────────────────────────────────
 * Node Retry Pattern: Reusable retry wrapper for graph nodes.
 *
 * ═══════════════════════════════════════════════════════════════════
 * THEORY: Retry Strategies
 * ═══════════════════════════════════════════════════════════════════
 *
 * Not all retries are equal. Different strategies suit different failures:
 *
 * 1. FIXED DELAY RETRY
 *    Wait the same time each retry: 1s → 1s → 1s
 *    Use for: Consistent rate-limited APIs
 *
 * 2. EXPONENTIAL BACKOFF
 *    Double the wait each time: 1s → 2s → 4s → 8s
 *    Use for: Overloaded services (gives them time to recover)
 *
 * 3. EXPONENTIAL BACKOFF WITH JITTER
 *    Add randomness to backoff: 1.2s → 2.7s → 3.9s
 *    Use for: Multiple clients hitting the same service
 *    (prevents "thundering herd" — all clients retrying at once)
 *
 * 4. IMMEDIATE RETRY
 *    Retry right away, no delay
 *    Use for: LLM output validation (re-prompt with correction)
 *
 * ═══════════════════════════════════════════════════════════════════
 *
 * Key Concepts:
 * - Create a reusable `withRetry` wrapper for any node function
 * - Supports configurable max retries and delay strategy
 * - Tracks retry count in state for observability
 * - The wrapper is transparent — wraps any (state) => update function
 *
 * Run: npx ts-node --esm src/14-error-handling/02-node-retry-decorator.ts
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
  temperature: 0.3,
  maxTokens: 200,
});

// ══════════════════════════════════════════════════════════════════
// Reusable Retry Wrapper
// ══════════════════════════════════════════════════════════════════
// This is a higher-order function — it takes a node function and
// returns a new function that retries on failure.

interface RetryOptions {
  maxRetries: number;
  delayMs: number;              // Base delay between retries
  backoff: "fixed" | "exponential";
  onRetry?: (error: Error, attempt: number) => void;
}

function withRetry<TState, TReturn>(
  nodeFn: (state: TState) => Promise<TReturn>,
  options: RetryOptions
): (state: TState) => Promise<TReturn> {
  return async (state: TState): Promise<TReturn> => {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= options.maxRetries + 1; attempt++) {
      try {
        return await nodeFn(state);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (attempt > options.maxRetries) {
          console.log(`  ❌ All ${options.maxRetries} retries exhausted`);
          throw lastError;
        }

        // Calculate delay
        const delay = options.backoff === "exponential"
          ? options.delayMs * Math.pow(2, attempt - 1)
          : options.delayMs;

        console.log(`  ⚠️  Attempt ${attempt} failed: ${lastError.message}`);
        console.log(`  ⏳ Retrying in ${delay}ms...`);

        options.onRetry?.(lastError, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  };
}

// ══════════════════════════════════════════════════════════════════
// Example: Flaky data processing pipeline with retry
// ══════════════════════════════════════════════════════════════════

const ProcessState = Annotation.Root({
  input: Annotation<string>(),
  fetchedData: Annotation<string>(),
  processedData: Annotation<string>(),
  retryInfo: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "No retries needed",
  }),
});

// Simulates a flaky external API
let callCount = 0;
async function fetchDataRaw(state: typeof ProcessState.State) {
  callCount++;
  console.log(`  🌐 Fetching data (call #${callCount})...`);

  // Fail the first 2 attempts to demonstrate retry
  if (callCount <= 2) {
    throw new Error(`API timeout (simulated failure #${callCount})`);
  }

  const res = await model.invoke(
    `Provide 3 interesting facts about: ${state.input}`
  );
  return {
    fetchedData: res.content as string,
    retryInfo: `Succeeded on attempt #${callCount}`,
  };
}

// Wrap with retry — exponential backoff, max 3 retries
const fetchData = withRetry(fetchDataRaw, {
  maxRetries: 3,
  delayMs: 500,      // 500ms → 1000ms → 2000ms
  backoff: "exponential",
  onRetry: (error, attempt) => {
    console.log(`  📊 Retry callback: attempt ${attempt}, error: ${error.message}`);
  },
});

async function processData(state: typeof ProcessState.State) {
  console.log("  ⚙️  Processing data...");
  const res = await model.invoke(
    `Summarize this in one sentence:\n\n${state.fetchedData}`
  );
  return { processedData: res.content as string };
}

// Build graph with retry-wrapped node
const graph = new StateGraph(ProcessState)
  .addNode("fetch", fetchData)      // Retry wrapper is transparent!
  .addNode("process", processData)
  .addEdge(START, "fetch")
  .addEdge("fetch", "process")
  .addEdge("process", END)
  .compile();

// ══════════════════════════════════════════════════════════════════
// Run
// ══════════════════════════════════════════════════════════════════

console.log("=== Node Retry Pattern ===\n");

const result = await graph.invoke({
  input: "TypeScript in AI development",
});

console.log("\n─── Results ───");
console.log("Processed:", result.processedData);
console.log("Retry info:", result.retryInfo);
