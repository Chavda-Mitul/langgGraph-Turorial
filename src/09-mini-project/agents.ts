/**
 * 09-mini-project/agents.ts
 * ──────────────────────────
 * Specialized agents using modern Command routing.
 *
 * Each agent returns a Command({ goto, update }) instead of
 * setting a nextAgent field. This is the recommended v1.x pattern.
 */

import { ChatGroq } from "@langchain/groq";
import { AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { Command } from "@langchain/langgraph";
import type { LangGraphRunnableConfig } from "@langchain/langgraph";
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
// Uses Command to route directly — no conditional edges needed.
export async function supervisorAgent(state: ContentState) {
  console.log("\n🎯 Supervisor: Evaluating pipeline...");

  let goto: string;
  let status = state.status;

  if (!state.research) {
    goto = "researcher";
    console.log("   → Dispatching to Researcher");
  } else if (!state.draft) {
    goto = "writer";
    console.log("   → Dispatching to Writer");
  } else if (!state.feedback) {
    goto = "reviewer";
    console.log("   → Dispatching to Reviewer");
  } else if (state.feedback.includes("APPROVED") || state.iteration >= 2) {
    goto = "finalizer";
    status = "finalizing";
    console.log("   → Dispatching to Finalizer");
  } else {
    goto = "writer";
    status = "revising";
    console.log("   → Revision needed, back to Writer");
  }

  // Command: route AND update state in one shot
  return new Command({
    goto,
    update: {
      status,
      log: [`[Supervisor] Routing to: ${goto}`],
      iteration: state.draft ? 1 : 0,
    },
  });
}

// ── Researcher Agent ───────────────────────────────────────────────
export async function researcherAgent(state: ContentState) {
  console.log("\n📚 Researcher: Gathering information...");

  const response = await modelWithTools.invoke([
    {
      role: "system",
      content: "You are a research specialist. Use the web_search tool to gather information. Search for 2-3 relevant queries, then summarize your findings.",
    },
    { role: "user", content: `Research this topic thoroughly: ${state.request}` },
  ]);

  const aiMsg = response as AIMessage;
  if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
    const toolNode = new ToolNode(allTools);
    const toolResults = await toolNode.invoke({ messages: [response] });

    const summaryRes = await model.invoke([
      { role: "system", content: "Summarize the research findings into a comprehensive brief." },
      { role: "user", content: `Topic: ${state.request}` },
      response,
      ...toolResults.messages,
    ]);

    return new Command({
      goto: "supervisor",
      update: {
        research: summaryRes.content as string,
        messages: [new AIMessage(`[Researcher] Research complete for: ${state.request}`)],
        log: ["[Researcher] Gathered data via web search"],
      },
    });
  }

  return new Command({
    goto: "supervisor",
    update: {
      research: response.content as string,
      messages: [new AIMessage("[Researcher] Research complete")],
      log: ["[Researcher] Generated research summary"],
    },
  });
}

// ── Writer Agent ───────────────────────────────────────────────────
export async function writerAgent(state: ContentState) {
  console.log("\n✍️  Writer: Crafting content...");

  const prompt = state.feedback && state.draft
    ? `Revise this draft based on the feedback.\n\nDRAFT:\n${state.draft}\n\nFEEDBACK:\n${state.feedback}\n\nWrite an improved version.`
    : `Write a well-structured article based on this research.\n\nTOPIC: ${state.request}\n\nRESEARCH:\n${state.research}\n\nWrite a clear, engaging article with intro, 2-3 key points, and conclusion. Under 300 words.`;

  const response = await model.invoke([
    { role: "system", content: "You are an expert content writer. Write clear, engaging content." },
    { role: "user", content: prompt },
  ]);

  return new Command({
    goto: "supervisor",
    update: {
      draft: response.content as string,
      feedback: "",
      messages: [new AIMessage(`[Writer] Draft ${state.iteration > 0 ? "revised" : "created"}`)],
      log: [`[Writer] ${state.iteration > 0 ? "Revised" : "Created"} draft`],
    },
  });
}

// ── Reviewer Agent ─────────────────────────────────────────────────
export async function reviewerAgent(state: ContentState) {
  console.log("\n📝 Reviewer: Evaluating draft...");

  const response = await model.invoke([
    {
      role: "system",
      content: `You are a strict but fair content reviewer.
Review for: clarity, accuracy, engagement, structure.
If good, respond with "APPROVED" + brief praise.
If not, give 2-3 actionable feedback points.`,
    },
    { role: "user", content: `Review this draft:\n\n${state.draft}` },
  ]);

  const feedback = response.content as string;
  const approved = feedback.toUpperCase().includes("APPROVED");
  console.log(`   Verdict: ${approved ? "✅ Approved" : "🔄 Needs revision"}`);

  return new Command({
    goto: "supervisor",
    update: {
      feedback,
      messages: [new AIMessage(`[Reviewer] ${approved ? "Approved" : "Revision needed"}`)],
      log: [`[Reviewer] ${approved ? "APPROVED" : "Requested revisions"}`],
    },
  });
}

// ── Finalizer Agent ────────────────────────────────────────────────
export async function finalizerAgent(
  state: ContentState,
  config: LangGraphRunnableConfig
) {
  console.log("\n🎨 Finalizer: Polishing content...");

  // Save the article topic to long-term memory (if store available)
  if (config.store) {
    const namespace = ["pipeline", "history"];
    await config.store.put(namespace, `article-${Date.now()}`, {
      topic: state.request,
      completedAt: new Date().toISOString(),
      iterations: state.iteration,
    });
    console.log("  💾 Saved article history to long-term memory");
  }

  const response = await model.invoke([
    {
      role: "system",
      content: "You are an editor. Polish this draft — fix grammar, improve flow, add a compelling title.",
    },
    { role: "user", content: state.draft },
  ]);

  return {
    finalContent: response.content as string,
    status: "completed",
    messages: [new AIMessage("[Finalizer] Content finalized!")],
    log: ["[Finalizer] Content polished and finalized"],
  };
}
