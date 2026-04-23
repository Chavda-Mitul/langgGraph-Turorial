/**
 * Model — Shared LLM Instance
 * ─────────────────────────────
 *
 * We create the ChatGroq model once and share it across all nodes.
 * This avoids creating multiple connections to the Groq API.
 *
 * "dotenv/config" loads the GROQ_API_KEY from your .env file.
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";

export const model = new ChatGroq({
    model: "llama-3.3-70b-versatile",
    temperature: 0.3,   // Low temperature = more factual, less creative
    maxTokens: 500,
});
