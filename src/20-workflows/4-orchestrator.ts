import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import z from "zod";

// 1. Initialize the Model (Set temperature to 0 for factual research)
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
});

// 2. Define the State with a REDUCER
// The reducer allows workerResults to accumulate data from multiple nodes
const GraphState = Annotation.Root({
  brand: Annotation<string>,
  tasks: Annotation<string[]>,
  workerResults: Annotation<string[]>({
    reducer: (current, next) => current.concat(next),
    default: () => [],
  }),
  finalReport: Annotation<string>,
});

const OrchestratorSchema = z.object({
  tasks: z.array(z.string()).length(3).describe("Exactly 3 specific research tasks for the brand"),
});

const orchestratorModel = model.withStructuredOutput(OrchestratorSchema);

// --- NODES ---

const orchestratorNode = async (state: typeof GraphState.State) => {
  console.log("Step 1: Orchestrating tasks for:", state.brand);
  const res = await orchestratorModel.invoke(`
    You are a research orchestrator for perfume brands.
    Brand: ${state.brand}

    Generate exactly 3 specific research tasks that will produce a comprehensive brand profile.
    Each task must be a clear, standalone instruction for a research agent.
    Cover: brand history, iconic products, and scent identity/DNA.
  `);
  console.log("Tasks generated:", res.tasks);
  return { tasks: res.tasks };
};
 
const historyWorker = async (state: typeof GraphState.State) => {
  console.log("Step 2a: Researching History...");
  const res = await model.invoke(state.tasks[0]);
  return { workerResults: [`[HISTORY]: ${res.content}`] };
};

const salesWorker = async (state: typeof GraphState.State) => {
  console.log("Step 2b: Researching Top Sellers...");
  const res = await model.invoke(state.tasks[1]);
  return { workerResults: [`[TOP SELLERS]: ${res.content}`] };
};

const dnaWorker = async (state: typeof GraphState.State) => {
  console.log("Step 2c: Researching Scent DNA...");
  const res = await model.invoke(state.tasks[2]);
  return { workerResults: [`[SCENT DNA]: ${res.content}`] };
};

const synthesizerNode = async (state: typeof GraphState.State) => {
  console.log("Step 3: Synthesizing final report...");
  const context = state.workerResults.join("\n\n");
  const res = await model.invoke(`
    Use the following research notes to create a professional brand profile for ${state.brand}.
    Structure it with headings for History, Iconic Scents, and Brand Philosophy.
    
    Research Notes:
    ${context}
  `);
  return { finalReport: res.content as string };
};

// --- GRAPH CONSTRUCTION ---

const workflow = new StateGraph(GraphState)
  .addNode("orchestrator", orchestratorNode)
  .addNode("history_worker", historyWorker)
  .addNode("sales_worker", salesWorker)
  .addNode("dna_worker", dnaWorker)
  .addNode("synthesizer", synthesizerNode)

  // Start with the Orchestrator
  .addEdge(START, "orchestrator")

  // FAN-OUT: Trigger all workers in parallel
  .addEdge("orchestrator", "history_worker")
  .addEdge("orchestrator", "sales_worker")
  .addEdge("orchestrator", "dna_worker")

  // FAN-IN: All workers lead to the Synthesizer
  .addEdge("history_worker", "synthesizer")
  .addEdge("sales_worker", "synthesizer")
  .addEdge("dna_worker", "synthesizer")

  .addEdge("synthesizer", END);

// --- EXECUTION ---

const app = workflow.compile();

const run = async () => {
  const result = await app.invoke({ brand: "Lattafa" });
  
  console.log("\n==========================================");
  console.log("FINAL BRAND REPORT");
  console.log("==========================================\n");
  console.log(result.finalReport);
};

run();