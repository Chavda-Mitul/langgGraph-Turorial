/**
 * 09-mini-project/graph.ts
 * ─────────────────────────
 * The graph definition: wires all agents together.
 *
 * Architecture (Supervisor Pattern):
 *
 *                    ┌──→ researcher ──┐
 *                    │                  │
 *   START → supervisor ──→ writer ──────→ supervisor (loop)
 *              ↑       │                  │
 *              │       └──→ reviewer ──────┘
 *              │
 *              └──→ finalizer ──→ END
 */

import {
  StateGraph,
  START,
  END,
  MemorySaver,
} from "@langchain/langgraph";
import { ContentPipelineState } from "./state.js";
import {
  supervisorAgent,
  researcherAgent,
  writerAgent,
  reviewerAgent,
  finalizerAgent,
} from "./agents.js";

// ── Router: reads nextAgent from state ─────────────────────────────
function routeNext(state: typeof ContentPipelineState.State): string {
  switch (state.nextAgent) {
    case "researcher": return "researcher";
    case "writer": return "writer";
    case "reviewer": return "reviewer";
    case "finalizer": return "finalizer";
    case "done": return "__end__";
    default: return "supervisor";
  }
}

// ── Build the graph ───────��────────────────────────────────────────
export function buildContentPipeline() {
  const graph = new StateGraph(ContentPipelineState)
    // Register all agent nodes
    .addNode("supervisor", supervisorAgent)
    .addNode("researcher", researcherAgent)
    .addNode("writer", writerAgent)
    .addNode("reviewer", reviewerAgent)
    .addNode("finalizer", finalizerAgent)

    // Entry: start with supervisor
    .addEdge(START, "supervisor")

    // Supervisor decides who goes next (conditional routing)
    .addConditionalEdges("supervisor", routeNext, [
      "researcher",
      "writer",
      "reviewer",
      "finalizer",
      "__end__",
    ])

    // All workers report back to supervisor
    .addEdge("researcher", "supervisor")
    .addEdge("writer", "supervisor")
    .addEdge("reviewer", "supervisor")

    // Finalizer ends the pipeline
    .addEdge("finalizer", END)

    // Compile with memory
    .compile({
      checkpointer: new MemorySaver(),
    });

  return graph;
}
