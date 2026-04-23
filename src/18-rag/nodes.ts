/**
 * Nodes — The Processing Steps of Our RAG Graph
 * ────────────────────────────────────────────────
 *
 * Each node is a function that:
 *   1. Reads from the state (the shared whiteboard)
 *   2. Does some work
 *   3. Returns ONLY the fields it wants to update
 *
 * Our graph has 3 nodes + 1 router:
 *
 *   [retriever]  → Searches ChromaDB for relevant document chunks
 *        ↓
 *   {router}     → Decides: are the retrieved chunks relevant?
 *      ↓    ↓
 *  [docAnswer] [llmAnswer]
 *      ↓    ↓
 *      END  END
 */

import type { Collection } from "chromadb";
import { model } from "./model.js";
import type { State } from "./schema.js";

// ─── SIMILARITY THRESHOLD ───
// ChromaDB returns a "distance" score (lower = more similar).
// If the best match has distance > this threshold, we consider it irrelevant.
// Tune this: lower = stricter matching, higher = more lenient.
const RELEVANCE_THRESHOLD = 1.5;

/**
 * We need the ChromaDB collection inside our nodes, but nodes only
 * receive `state` as input. Solution: create the nodes using a factory
 * function that "closes over" the collection.
 *
 * This is a common pattern in LangGraph when nodes need external resources.
 */
export function createNodes(collection: Collection) {

    // ─── Node 1: RETRIEVER ───
    // Searches ChromaDB for chunks that are similar to the user's question.
    // Think of it like a librarian finding the most relevant pages.
    const retrieverNode = async (state: State) => {
        console.log(`\n[Retriever] Searching docs for: "${state.question}"`);

        // Query ChromaDB — it converts the question to a vector and finds
        // the closest matching document chunks
        const results = await collection.query({
            queryTexts: [state.question],   // What to search for
            nResults: 3,                     // Return top 3 matches
        });

        // Extract the matched text chunks and their distances
        const documents = results.documents?.[0] || [];
        const distances = results.distances?.[0] || [];

        console.log(`[Retriever] Found ${documents.length} chunks`);
        distances.forEach((dist, i) => {
            console.log(`  Chunk ${i + 1}: distance=${dist.toFixed(3)} | "${documents[i]?.slice(0, 60)}..."`);
        });

        // Return the chunks as context — the next node will use these
        return {
            context: documents.filter((d): d is string => d !== null),
        };
    };

    // ─── Router: SHOULD WE USE DOCS OR LLM? ───
    // This is NOT a node — it's a function that returns a string
    // telling the graph which node to go to next.
    //
    // Same pattern as checkContentLength in module 17!
    const routeByRelevance = (state: State): string => {
        // If we have no context at all, go to LLM
        if (!state.context || state.context.length === 0) {
            console.log("[Router] No context found → using LLM");
            return "no_context";
        }

        // Check if any chunk is relevant enough
        // We re-query to get distances (or we could store them in state)
        // For simplicity, we check if context has meaningful content
        const hasContent = state.context.some(chunk => chunk.length > 30);

        if (hasContent) {
            console.log("[Router] Relevant context found → answering from docs");
            return "has_context";
        } else {
            console.log("[Router] Context not relevant → using LLM");
            return "no_context";
        }
    };

    // ─── Node 2: DOC ANSWER ───
    // Answers the question using ONLY the retrieved document context.
    // The system prompt is critical here — it tells the LLM to stick to the docs.
    const docAnswerNode = async (state: State) => {
        console.log("[DocAnswer] Generating answer from company documents...");

        // Join all retrieved chunks into one context block
        const contextText = state.context.join("\n\n---\n\n");

        const response = await model.invoke([
            {
                role: "system",
                content: `You are a helpful company assistant. Answer the user's question based ONLY on the following company document excerpts. If the excerpts don't contain enough information to fully answer, say what you can from the docs and mention that the information may be incomplete.

COMPANY DOCUMENTS:
${contextText}`,
            },
            {
                role: "user",
                content: state.question,
            },
        ]);

        return {
            answer: response.content as string,
            source: "document",   // Mark that this answer came from docs
        };
    };

    // ─── Node 3: LLM ANSWER ───
    // Fallback — answers using the LLM's general knowledge.
    // Used when no relevant company docs are found.
    const llmAnswerNode = async (state: State) => {
        console.log("[LLMAnswer] No relevant docs found. Using general LLM knowledge...");

        const response = await model.invoke([
            {
                role: "system",
                content: "You are a helpful assistant. Answer the user's question using your general knowledge. Be concise and helpful.",
            },
            {
                role: "user",
                content: state.question,
            },
        ]);

        return {
            answer: response.content as string,
            source: "llm",   // Mark that this answer came from general LLM
        };
    };

    return { retrieverNode, routeByRelevance, docAnswerNode, llmAnswerNode };
}
