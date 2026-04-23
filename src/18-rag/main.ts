/**
 * Main — Entry Point for the RAG Company Bot
 * ─────────────────────────────────────────────
 *
 * This file ties everything together:
 *   1. Ingests documents from the docs/ folder into ChromaDB
 *   2. Builds the RAG graph
 *   3. Runs an interactive question loop
 *
 * Usage:
 *   1. Put your company PDF files in src/18-rag/docs/
 *   2. Run: npx tsx src/18-rag/main.ts
 *   3. Ask questions! Type "exit" to quit.
 *
 * How it works:
 *   Your question → Retriever searches docs → Router decides path
 *     → If relevant docs found: answers from docs (source: "document")
 *     → If no relevant docs:   answers from LLM  (source: "llm")
 */

import { ingestDocuments } from "./ingest.js";
import { buildRAGWorkflow } from "./workflow.js";
import * as readline from "node:readline/promises";

async function main() {
    console.log("╔══════════════════════════════════════════╗");
    console.log("║     🤖 RAG Company Bot — Module 18       ║");
    console.log("╚══════════════════════════════════════════╝\n");

    // ─── Step 1: Ingest documents ───
    console.log("--- 📄 Phase 1: Loading Documents ---\n");
    const collection = await ingestDocuments();

    // ─── Step 2: Build the graph ───
    console.log("\n--- 🔧 Phase 2: Building RAG Graph ---\n");
    const app = buildRAGWorkflow(collection);
    console.log("Graph ready!\n");

    // ─── Step 3: Interactive question loop ───
    console.log("--- 💬 Phase 3: Ask Questions ---");
    console.log('Type your question and press Enter. Type "exit" to quit.\n');

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    while (true) {
        const question = await rl.question("You: ");

        // Exit condition
        if (question.toLowerCase() === "exit") {
            console.log("\nGoodbye! 👋");
            break;
        }

        // Skip empty input
        if (!question.trim()) continue;

        // Run the question through our RAG graph
        const result = await app.invoke({ question });

        // Display the answer with its source
        console.log(`\n📌 Source: ${result.source === "document" ? "Company Documents 📄" : "General LLM Knowledge 🧠"}`);
        console.log(`\nBot: ${result.answer}\n`);
    }

    rl.close();
}

main();
