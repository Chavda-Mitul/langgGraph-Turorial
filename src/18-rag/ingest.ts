/**
 * Ingest — Load Documents into ChromaDB
 * ────────────────────────────────────────
 *
 * This file handles the "preparation" phase of RAG:
 *   1. Read all PDF files from the docs/ folder
 *   2. Split them into small chunks (paragraphs)
 *   3. Store them in ChromaDB with embeddings
 *
 * ChromaDB automatically generates embeddings using its default
 * embedding function, so we don't need a separate embedding model.
 *
 * HOW CHUNKING WORKS:
 *   Imagine a 10-page PDF. You can't send all 10 pages to the LLM
 *   every time someone asks a question. Instead, we split it into
 *   small pieces (chunks) of ~500 characters each. When a user asks
 *   a question, we only retrieve the 3-4 most relevant chunks.
 *
 *   "What is our refund policy?" → only retrieves the chunk about refunds,
 *   not the entire employee handbook.
 */

import fs from "node:fs";
import path from "node:path";
import {PDFParse} from "pdf-parse";
import { ChromaClient } from "chromadb";

// ─── Configuration ───
const CHUNK_SIZE = 500;      // Max characters per chunk
const CHUNK_OVERLAP = 50;    // Characters to overlap between chunks (prevents cutting mid-sentence)
const COLLECTION_NAME = "company_docs";

/**
 * Split text into overlapping chunks.
 *
 * Why overlap? Imagine this text gets split right at "refund policy":
 *   Chunk 1: "...customers can request a"
 *   Chunk 2: "refund policy allows 30 days..."
 *
 * With overlap, Chunk 1 would also contain "refund policy allows",
 * so the search can find it regardless of where the split happened.
 */
function splitIntoChunks(text: string): string[] {
    const chunks: string[] = [];
    let start = 0;

    while (start < text.length) {
        const end = start + CHUNK_SIZE;
        const chunk = text.slice(start, end).trim();

        // Only keep chunks with meaningful content (not just whitespace)
        if (chunk.length > 20) {
            chunks.push(chunk);
        }

        // Move forward, but step back by CHUNK_OVERLAP for overlap
        start = end - CHUNK_OVERLAP;
    }

    return chunks;
}

/**
 * Read a single PDF file and return its text content.
 */
async function readPDF(filePath: string): Promise<string> {
    const buffer = fs.readFileSync(filePath);
    const data = await new PDFParse(buffer).getText();
    return data.text;
}

/**
 * Main ingestion function:
 *   1. Scans docs/ folder for PDFs
 *   2. Reads and chunks each PDF
 *   3. Stores all chunks in ChromaDB
 *
 * Returns the ChromaDB collection so the graph can use it for retrieval.
 */
export async function ingestDocuments() {
    const docsDir = path.join(import.meta.dirname, "docs");

    // Step 1: Find all PDF files in the docs/ folder
    if (!fs.existsSync(docsDir)) {
        throw new Error(`Docs folder not found: ${docsDir}\nCreate it and add your PDF files there.`);
    }

    const pdfFiles = fs.readdirSync(docsDir).filter(f => f.endsWith(".pdf"));

    if (pdfFiles.length === 0) {
        throw new Error(`No PDF files found in ${docsDir}\nAdd your company documents there.`);
    }

    console.log(`[Ingest] Found ${pdfFiles.length} PDF file(s): ${pdfFiles.join(", ")}`);

    // Step 2: Read and chunk all PDFs
    const allChunks: string[] = [];
    const allMetadata: Array<{ source: string; chunkIndex: number }> = [];

    for (const file of pdfFiles) {
        const filePath = path.join(docsDir, file);
        console.log(`[Ingest] Reading: ${file}...`);

        const text = await readPDF(filePath);
        const chunks = splitIntoChunks(text);

        console.log(`[Ingest]   → ${chunks.length} chunks created from ${file}`);

        for (let i = 0; i < chunks.length; i++) {
            allChunks.push(chunks[i]);
            allMetadata.push({ source: file, chunkIndex: i });
        }
    }

    console.log(`[Ingest] Total chunks to store: ${allChunks.length}`);

    // Step 3: Store in ChromaDB
    // ChromaDB runs in-memory here (no server needed)
    const client = new ChromaClient();

    // Delete old collection if it exists (fresh start each time)
    try {
        await client.deleteCollection({ name: COLLECTION_NAME });
    } catch {
        // Collection didn't exist — that's fine
    }

    // Create a new collection — ChromaDB will auto-generate embeddings
    const collection = await client.getOrCreateCollection({
        name: COLLECTION_NAME,
    });

    // Add all chunks to the collection
    // Each chunk gets: an ID, the text (document), and metadata (which file it came from)
    await collection.add({
        ids: allChunks.map((_, i) => `chunk-${i}`),
        documents: allChunks,
        metadatas: allMetadata,
    });

    console.log(`[Ingest] ✅ All documents stored in ChromaDB collection "${COLLECTION_NAME}"`);

    return collection;
}
