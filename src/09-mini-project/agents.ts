/**
 * 09-mini-project/agents.ts
 * ──────────────────────────
 * Specialized agents: Supervisor, Researcher, Writer, Reviewer.
 *
 * Each agent is a node in the graph that processes state
 * and returns state updates.
 */

import { ChatGroq } from "@langchain/groq";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ContentState } from "./state.js";
import { allTools } from "./tools.js";

// Shared model
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.5,
  maxTokens: 500,
});

const modelWithTools = model.bindTools(allTools);

// ── Supervisor Agent ───────────────────────────────────────────────
// Decides which agent should work next based on current state.
export async function supervisorAgent(state: ContentState) {
  console.log("\n🎯 Supervisor: Evaluating pipeline...");

  let nextAgent: string;
  let status = state.status;

  if (!state.research) {
    nextAgent = "researcher";
    console.log("   → Dispatching to Researcher");
  } else if (!state.draft) {
    nextAgent = "writer";
    console.log("   → Dispatching to Writer");
  } else if (!state.feedback) {
    nextAgent = "reviewer";
    console.log("   → Dispatching to Reviewer");
  } else if (state.feedback.includes("APPROVED") || state.iteration >= 2) {
    nextAgent = "finalizer";
    status = "finalizing";
    console.log("   → Dispatching to Finalizer");
  } else {
    nextAgent = "writer";
    status = "revising";
    console.log("   → Revision needed, back to Writer");
  }

  return {
    nextAgent,
    status,
    log: [`[Supervisor] Routing to: ${nextAgent}`],
    iteration: state.draft ? 1 : 0,
  };
}

// ── Researcher Agent ───────────────────────────────────────────────
// Gathers information using tools.
export async function researcherAgent(state: ContentState) {
  console.log("\n📚 Researcher: Gathering information...");

  // Use the model with tools to research
  const response = await modelWithTools.invoke([
    {
      role: "system",
      content: "You are a research specialist. Use the web_search tool to gather information. Search for 2-3 relevant queries, then summarize your findings.",
    },
    new HumanMessage(`Research this topic thoroughly: ${state.request}`),
  ]);

  // Check if tools were called
  const aiMsg = response as AIMessage;
  if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
    // Execute tools
    const toolNode = new ToolNode(allTools);
    const toolResults = await toolNode.invoke({
      messages: [response],
    });

    // Summarize with tool results
    const summaryRes = await model.invoke([
      {
        role: "system",
        content: "Summarize the research findings into a comprehensive brief.",
      },
      new HumanMessage(`Topic: ${state.request}`),
      response,
      ...toolResults.messages,
    ]);

    return {
      research: summaryRes.content as string,
      messages: [new AIMessage(`[Researcher] Research complete for: ${state.request}`)],
      log: ["[Researcher] Gathered data via web search"],
      nextAgent: "supervisor",
    };
  }

  return {
    research: response.content as string,
    messages: [new AIMessage(`[Researcher] Research complete`)],
    log: ["[Researcher] Generated research summary"],
    nextAgent: "supervisor",
  };
}

// ── Writer Agent ───────────────────────────────────────────────────
// Creates or revises content based on research and feedback.
export async function writerAgent(state: ContentState) {
  console.log("\n✍️  Writer: Crafting content...");

  let prompt: string;
  if (state.feedback && state.draft) {
    prompt = `Revise this draft based on the feedback.

ORIGINAL DRAFT:
${state.draft}

FEEDBACK:
${state.feedback}

Write an improved version. Keep it focused and engaging.`;
  } else {
    prompt = `Write a well-structured article based on this research.

TOPIC: ${state.request}

RESEARCH:
${state.research}

Write a clear, engaging article with an introduction, 2-3 key points, and a conclusion. Keep it under 300 words.`;
  }

  const response = await model.invoke([
    {
      role: "system",
      content: "You are an expert content writer. Write clear, engaging, well-structured content.",
    },
    new HumanMessage(prompt),
  ]);

  return {
    draft: response.content as string,
    feedback: "", // Clear old feedback for fresh review
    messages: [new AIMessage(`[Writer] Draft ${state.iteration > 0 ? "revised" : "created"}`)],
    log: [`[Writer] ${state.iteration > 0 ? "Revised" : "Created"} draft`],
    nextAgent: "supervisor",
  };
}

// ── Reviewer Agent ─────────────────────────────────────────────────
// Reviews the draft and provides feedback or approval.
export async function reviewerAgent(state: ContentState) {
  console.log("\n📝 Reviewer: Evaluating draft...");

  const response = await model.invoke([
    {
      role: "system",
      content: `You are a strict but fair content reviewer.
Review the draft for: clarity, accuracy, engagement, and structure.
If the draft is good, respond with "APPROVED" followed by brief praise.
If it needs improvement, provide specific, actionable feedback (2-3 points max).`,
    },
    new HumanMessage(`Review this draft:\n\n${state.draft}`),
  ]);

  const feedback = response.content as string;
  const approved = feedback.toUpperCase().includes("APPROVED");

  console.log(`   Verdict: ${approved ? "✅ Approved" : "🔄 Needs revision"}`);

  return {
    feedback,
    messages: [new AIMessage(`[Reviewer] ${approved ? "Approved" : "Revision needed"}`)],
    log: [`[Reviewer] ${approved ? "APPROVED" : "Requested revisions"}`],
    nextAgent: "supervisor",
  };
}

// ── Finalizer Agent ────────────────────────────────────────────────
// Polishes the final content.
export async function finalizerAgent(state: ContentState) {
  console.log("\n🎨 Finalizer: Polishing content...");

  const response = await model.invoke([
    {
      role: "system",
      content: "You are an editor. Polish this draft — fix any remaining grammar issues, improve flow, and add a compelling title. Return the final version with the title.",
    },
    new HumanMessage(state.draft),
  ]);

  return {
    finalContent: response.content as string,
    status: "completed",
    messages: [new AIMessage("[Finalizer] Content finalized!")],
    log: ["[Finalizer] Content polished and finalized"],
    nextAgent: "done",
  };
}
