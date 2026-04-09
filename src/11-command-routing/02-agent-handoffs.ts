/**
 * 11-command-routing/02-agent-handoffs.ts
 * ────────────────────────────────────────
 * Agent Handoffs: Agents that transfer control to other agents.
 *
 * Key Concepts:
 * - createHandoffTool(): Creates a tool that transfers to another agent
 * - Command.PARENT: Route in the PARENT graph (not the current subgraph)
 * - getCurrentTaskInput(): Access the current graph state inside tools
 * - Multi-agent with ReAct agents + handoff tools
 * - Each agent is a subgraph node in a parent graph
 *
 * NOTE: LangGraph now provides a BUILT-IN createHandoffTool in the prebuilt
 * module. We build our own here for educational purposes — to understand HOW
 * handoffs work under the hood. In production, prefer the built-in version.
 *
 * This is the RECOMMENDED pattern for production multi-agent systems.
 *
 * Run: npx ts-node --esm src/11-command-routing/02-agent-handoffs.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ToolMessage, HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  StateGraph,
  MessagesAnnotation,
  Command,
  START,
  END,
  getCurrentTaskInput,
} from "@langchain/langgraph";

// ── 1. Handoff tool factory (custom, for learning) ────────────────
// This creates a tool that any agent can use to transfer to another agent.
// In production, you can use the built-in version:
//   import { createHandoffTool } from "@langchain/langgraph/prebuilt";

function createHandoffTool(agentName: string, description?: string) {
  const toolName = `transfer_to_${agentName}`;

  return tool(
    async (_, config) => {
      const toolMessage = new ToolMessage({
        content: `Successfully transferred to ${agentName}`,
        name: toolName,
        tool_call_id: config.toolCall.id,
      });

      // Get current state and route to the target agent in the PARENT graph
      const state = getCurrentTaskInput() as (typeof MessagesAnnotation)["State"];
      return new Command({
        goto: agentName,
        update: { messages: state.messages.concat(toolMessage) },
        graph: Command.PARENT, // Route in the parent, not current subgraph!
      });
    },
    {
      name: toolName,
      description: description || `Transfer to ${agentName}`,
      schema: z.object({}),
    }
  );
}

// ── 2. Domain tools ────────────────────────────────────────────────
const lookupFlight = tool(
  async ({ from, to }) => `Flight found: ${from} → ${to}, $350, departs 2:30 PM`,
  {
    name: "lookup_flight",
    description: "Look up available flights",
    schema: z.object({
      from: z.string().describe("Departure city"),
      to: z.string().describe("Destination city"),
    }),
  }
);

const bookHotel = tool(
  async ({ city, nights }) => `Hotel booked: ${city}, ${nights} nights at Grand Hotel, $120/night`,
  {
    name: "book_hotel",
    description: "Book a hotel in a city",
    schema: z.object({
      city: z.string().describe("City name"),
      nights: z.number().describe("Number of nights"),
    }),
  }
);

// ── 3. Create specialized agents ───────────────────────────────────
const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.3,
  maxTokens: 300,
});

// Flight agent — can hand off to hotel agent
const flightAgent = createReactAgent({
  llm,
  tools: [lookupFlight, createHandoffTool("hotel_agent", "Transfer to hotel booking agent")],
  prompt: "You are a flight booking assistant. Help users find and book flights. If they need a hotel, transfer them to the hotel agent.",
  name: "flight_agent",
});

// Hotel agent — can hand off to flight agent
const hotelAgent = createReactAgent({
  llm,
  tools: [bookHotel, createHandoffTool("flight_agent", "Transfer to flight booking agent")],
  prompt: "You are a hotel booking assistant. Help users book hotels. If they need a flight, transfer them to the flight agent.",
  name: "hotel_agent",
});

// ── 4. Parent graph — connects agents ──────────────────────────────
const travelGraph = new StateGraph(MessagesAnnotation)
  .addNode("flight_agent", flightAgent, {
    ends: ["hotel_agent", END],  // Can hand off to hotel or end
  })
  .addNode("hotel_agent", hotelAgent, {
    ends: ["flight_agent", END], // Can hand off to flight or end
  })
  .addEdge(START, "flight_agent")
  .compile();

// ── 5. Test: agent handoff in action ───────────────────────────────
console.log("=== Agent Handoffs ===\n");

const result = await travelGraph.invoke({
  messages: [
    new HumanMessage(
      "I need a flight from Delhi to Mumbai, and also book a hotel in Mumbai for 3 nights."
    ),
  ],
});

console.log("\n--- Conversation ---");
for (const msg of result.messages) {
  const role = msg.getType();
  if (role === "human") {
    console.log(`\n👤 User: ${msg.content}`);
  } else if (role === "ai" && msg.content) {
    console.log(`\n🤖 Agent: ${msg.content}`);
  } else if (role === "tool") {
    console.log(`\n🔧 Tool: ${msg.content}`);
  }
}
