/**
 * Workflow — Wiring the RAG Graph Together
 * ──────────────────────────────────────────
 *
 * This is where we connect all the nodes with edges to form the graph:
 *
 *   START → retriever → {router} → docAnswer → END
 *                           ↓
 *                        llmAnswer → END
 *
 * The router is a conditional edge — it looks at the state and decides
 * which path to take, just like module 17's checkContentLength.
 *
 * We use a factory function because the nodes need the ChromaDB
 * collection, which only exists after ingestion.
 */

import { END, START, StateGraph } from "@langchain/langgraph";
import type { Collection } from "chromadb";
import { RAGState } from "./schema.js";
import { createNodes } from "./nodes.js";

export function buildRAGWorkflow(collection: Collection) {
    // Create all nodes with access to the ChromaDB collection
    const { retrieverNode, routeByRelevance, docAnswerNode, llmAnswerNode } = createNodes(collection);

    // Build the graph — same pattern as every other module!
    const workflow = new StateGraph(RAGState)

        // Step 1: Register all nodes
        .addNode("retriever", retrieverNode)
        .addNode("docAnswer", docAnswerNode)
        .addNode("llmAnswer", llmAnswerNode)

        // Step 2: Connect with edges

        // START → always go to retriever first (search the docs)
        .addEdge(START, "retriever")

        // Retriever → conditional: check if docs are relevant
        // This is the "brain" — it decides the path based on state
        .addConditionalEdges("retriever", routeByRelevance, {
            "has_context": "docAnswer",    // Good context → answer from docs
            "no_context": "llmAnswer",     // No context → fall back to LLM
        })

        // Both answer nodes → END (the graph is done)
        .addEdge("docAnswer", END)
        .addEdge("llmAnswer", END);

    return workflow.compile();
}
