/**
 * 06-human-in-the-loop/02-tool-approval.ts
 * ──────────────────────────────────────────
 * Tool Approval: Require human approval before executing tools.
 *
 * Key Concepts:
 * - interruptBefore: ["tools"] — pause before tool execution
 * - Inspect pending tool calls before they run
 * - Resume to allow execution, or modify state to change/skip tools
 * - Real-world use: prevent dangerous actions, review API calls
 *
 * Run: npx ts-node --esm src/06-human-in-the-loop/02-tool-approval.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  MemorySaver,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// ── 1. Sensitive tools (need approval!) ────────────────────────────
const deleteFile = tool(
  async ({ filename }) => `Deleted file: ${filename}`,
  {
    name: "delete_file",
    description: "Delete a file from the system (dangerous!)",
    schema: z.object({
      filename: z.string().describe("File to delete"),
    }),
  }
);

const sendEmail = tool(
  async ({ to, subject }) => `Email sent to ${to}: "${subject}"`,
  {
    name: "send_email",
    description: "Send an email to someone",
    schema: z.object({
      to: z.string().describe("Recipient email"),
      subject: z.string().describe("Email subject"),
    }),
  }
);

const tools = [deleteFile, sendEmail];

// ── 2. Agent ───────────────────────────────────────────────────────
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  maxTokens: 256,
}).bindTools(tools);

async function agent(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke([
    { role: "system", content: "You are a helpful assistant with access to file and email tools." },
    ...state.messages,
  ]);
  return { messages: [response] };
}

function shouldContinue(state: typeof MessagesAnnotation.State): string {
  const lastMsg = state.messages[state.messages.length - 1] as AIMessage;
  return lastMsg.tool_calls?.length ? "tools" : END;
}

// ── 3. Compile with interruptBefore ────────────────────────────────
const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", agent)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, ["tools", END])
  .addEdge("tools", "agent")
  .compile({
    checkpointer: new MemorySaver(),
    interruptBefore: ["tools"], // ← Pause BEFORE executing tools!
  });

// ── 4. Run: graph pauses before tool execution ─────────────────────
const config = { configurable: { thread_id: "tool-approval-1" } };

console.log("=== Tool Approval Workflow ===\n");

// Step 1: Agent decides to use a tool
console.log("--- Step 1: Agent plans action ---");
await graph.invoke(
  { messages: [new HumanMessage("Send an email to mansi@example.com with subject 'Meeting Tomorrow'")] },
  config
);

// Step 2: Check what tool the agent wants to call
const state = await graph.getState(config);
console.log("Graph paused. Next:", state.next);

const lastMsg = state.values.messages[state.values.messages.length - 1] as AIMessage;
if (lastMsg.tool_calls?.length) {
  console.log("\nPending tool calls:");
  lastMsg.tool_calls.forEach(tc => {
    console.log(`  📧 ${tc.name}(${JSON.stringify(tc.args)})`);
  });
}

// Step 3: Resume (approve the tool call)
console.log("\n--- Step 2: Human approves, resuming ---");
const result = await graph.invoke(null, config); // null = continue as-is

console.log("\nFinal:", result.messages[result.messages.length - 1].content);
