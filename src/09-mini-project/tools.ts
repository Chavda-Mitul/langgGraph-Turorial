/**
 * 09-mini-project/tools.ts
 * ─────────────────────────
 * Tools available to the research agent.
 */

import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Simulated web search tool
export const webSearch = tool(
  async ({ query }) => {
    // Simulated search results
    const results: Record<string, string> = {
      "ai agents": "AI agents are autonomous systems that use LLMs to perceive, reason, and act. Key frameworks: LangGraph, CrewAI, AutoGen. Trends: multi-agent systems, tool use, and human-in-the-loop patterns.",
      "langgraph": "LangGraph is a graph-based orchestration framework by LangChain. It enables stateful, multi-actor AI applications with cycles, persistence, and streaming. Used by enterprises for complex agent workflows.",
      "multi-agent": "Multi-agent systems coordinate multiple specialized AI agents. Patterns include: supervisor (coordinator), hierarchical (layered), and collaborative (peer-to-peer). Benefits: specialization, parallel processing, and robustness.",
      "typescript ai": "TypeScript is increasingly popular for AI development. Benefits: type safety, great tooling, runs on Node.js. Key libraries: LangChain.js, LangGraph.js, Vercel AI SDK.",
    };

    const key = Object.keys(results).find(k =>
      query.toLowerCase().includes(k)
    );
    return key
      ? results[key]
      : `Search results for "${query}": This is a rapidly evolving field with many emerging patterns and best practices.`;
  },
  {
    name: "web_search",
    description: "Search the web for information about AI topics",
    schema: z.object({
      query: z.string().describe("Search query"),
    }),
  }
);

// Fact checker tool
export const factChecker = tool(
  async ({ claim }) => {
    // Simulated fact checking
    const length = claim.length;
    if (length > 100) return "✅ Claim appears well-supported by multiple sources.";
    if (length > 50) return "⚠️ Claim is partially supported. Consider adding citations.";
    return "❓ Claim needs verification. Insufficient context to verify.";
  },
  {
    name: "fact_checker",
    description: "Verify a factual claim",
    schema: z.object({
      claim: z.string().describe("The claim to verify"),
    }),
  }
);

export const allTools = [webSearch, factChecker];
