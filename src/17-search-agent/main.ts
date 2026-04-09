import { MemorySaver } from "@langchain/langgraph";
import { workflow } from "./workflow.js";
import { threadId } from "worker_threads";

// 1. Initialize the memory "database"
const checkpointer = new MemorySaver();

// 2. Compile with the checkpointer AND an interrupt
const app = workflow.compile({
    checkpointer,
    // This tells the graph: "Pause every time you finish the researcher node"
    interruptBefore: ['editor']
});

async function main() {

    const config = {configurable: {thread_id: 'user-1'}};
    // --- FIRST RUN ---
    console.log("--- 🚀 PHASE 1 Starting Research ---");

    // The first invoke starts the graph and stops BEFORE 'editor'
    const snapshot = await app.invoke({topic: "generative ai"},config);

    // At this point, snapshot.rawContent will exist, 
    // but snapshot.polishedContent will be empty/undefined!
    console.log("LOG: Researcher finished. Graph is now PAUSED.");
    console.log("Raw Research found:", snapshot.rawContent);

    console.log("\n-----------------------------------------");
    console.log("WAITING FOR HUMAN APPROVAL...");
    console.log("-----------------------------------------\n");

    // 2. THE RESUME
    // To continue, we call invoke again.
    // We pass 'null' as the first argument because the state is already saved in memory.
    console.log("--- 🚀 PHASE 2: Resuming to Editor & Summary ---");
    const finalResult = await app.invoke(null, config);

    console.log("\n--- ✅ Final Workflow Result ---");
    console.log("Polished:", finalResult.polishedContent);
    console.log("Summary:", finalResult.summary);
}

main();