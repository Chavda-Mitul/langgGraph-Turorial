/**
 * 09-mini-project/graph.ts
 * ─────────────────────────
 * The graph definition using modern Command-based routing.
 *
 * Architecture (Supervisor Pattern with Command routing):
 *
 *                    ┌──→ researcher ──┐
 *                    │                  │
 *   START → supervisor ──→ writer ──────→ supervisor (loop)
 *              ↑       │                  │
 *              │       └──→ reviewer ──────┘
 *              │
 *              └──→ finalizer ──→ END
 *
 * Key difference from old pattern:
 * - No addConditionalEdges() — agents return Command({ goto }) directly
 * - addNode with { ends: [...] } declares possible destinations
 * - InMemoryStore for cross-session article history
 */

import {
  StateGraph,
  START,
  END,
  MemorySaver,
  InMemoryStore,
} from "@langchain/langgraph";
import { ContentPipelineState } from "./state.js";
import {
  supervisorAgent,
  researcherAgent,
  writerAgent,
  reviewerAgent,
  finalizerAgent,
} from "./agents.js";

// ── Build the graph ────────────────────────────────────────────────
export function buildContentPipeline() {
  const graph = new StateGraph(ContentPipelineState)
    // Supervisor uses Command to route — declare possible destinations with `ends`
    .addNode("supervisor", supervisorAgent, {
      ends: ["researcher", "writer", "reviewer", "finalizer"],
    })
    // Workers use Command to go back to supervisor
    .addNode("researcher", researcherAgent, { ends: ["supervisor"] })
    .addNode("writer", writerAgent, { ends: ["supervisor"] })
    .addNode("reviewer", reviewerAgent, { ends: ["supervisor"] })
    // Finalizer returns plain state update — uses normal edge to END
    .addNode("finalizer", finalizerAgent)

    // Entry point
    .addEdge(START, "supervisor")
    // Only finalizer needs a static edge (others use Command)
    .addEdge("finalizer", END)

    // Compile with both short-term and long-term memory
    .compile({
      checkpointer: new MemorySaver(),    // Thread-level state
      store: new InMemoryStore(),          // Cross-thread article history
    });

  return graph;
}
