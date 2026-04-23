/**
 * Schema — The State Definition for our RAG Pipeline
 * ─────────────────────────────────────────────────────
 *
 * This defines ALL the data that flows through our graph.
 * Every node can read from and write to these fields.
 *
 * Think of it as a shared whiteboard:
 *   - "question"  → what the user asked
 *   - "context"   → relevant chunks we found from company docs
 *   - "answer"    → the final answer we generated
 *   - "source"    → did the answer come from docs or general LLM knowledge?
 */

import { Annotation } from "@langchain/langgraph";

export const RAGState = Annotation.Root({
    // The user's question (input)
    question: Annotation<string>,

    // Retrieved document chunks from ChromaDB (filled by retriever node)
    // This will be an array of text strings from your company docs
    context: Annotation<string[]>({
        reducer: (_old, newVal) => newVal,   // always replace with latest retrieval
        default: () => [],
    }),

    // The final answer (filled by either doc-answer or llm-answer node)
    answer: Annotation<string>,

    // Where did the answer come from? "document" or "llm"
    // This helps the user know if the answer is from company docs or general knowledge
    source: Annotation<string>,
});

// Extract the TypeScript type so we can use it in our node functions
export type State = typeof RAGState.State;
