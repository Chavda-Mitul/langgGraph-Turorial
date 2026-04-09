/**
 * 08-prebuilt-agents/01-react-agent.ts
 * ─────────────────────────────────────
 * createReactAgent: Build a full agent in ONE function call.
 *
 * Key Concepts:
 * - createReactAgent(): Prebuilt agent that handles the entire loop
 * - Automatically creates: model node → tool check → tool node → loop
 * - Supports: memory (checkpointer), tool approval, system prompt
 * - `prompt` parameter: pass a string OR a function for dynamic prompts
 *   (replaces older `stateModifier` — both work, but `prompt` is preferred)
 * - Use this when you want a standard agent without custom graph logic
 * - Use custom StateGraph when you need special routing or multi-agent
 *
 * Under the hood, createReactAgent builds this graph:
 *   START → agent → (tool_calls?) → tools → agent → ... → END
 *
 * Run: npx ts-node --esm src/08-prebuilt-agents/01-react-agent.ts
 */

import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

// ── 1. Define tools ────────────────────────────────────────────────
const getWeather = tool(
  async ({ city }) => {
    const weather: Record<string, string> = {
      "mumbai": "32°C, Humid and partly cloudy",
      "delhi": "28°C, Sunny with mild winds",
      "bangalore": "24°C, Pleasant with light rain",
      "san francisco": "18°C, Foggy as usual",
    };
    return weather[city.toLowerCase()] || `Weather data not available for ${city}`;
  },
  {
    name: "get_weather",
    description: "Get current weather for a city",
    schema: z.object({
      city: z.string().describe("City name"),
    }),
  }
);

const getNews = tool(
  async ({ topic }) => {
    return `Latest news about ${topic}: AI continues to transform industries. New frameworks make building agents easier than ever. [Simulated news]`;
  },
  {
    name: "get_news",
    description: "Get latest news about a topic",
    schema: z.object({
      topic: z.string().describe("News topic"),
    }),
  }
);

// ── 2. Create ReAct agent ──────────────────────────────────────────
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.3,
  maxTokens: 300,
});

const agent = createReactAgent({
  llm: model,
  tools: [getWeather, getNews],
  // Optional: system prompt to customize behavior
  // `prompt` replaces the older `stateModifier` (both still work)
  // Can be a string OR a function: (state) => BaseMessageLike[]
  prompt: "You are a helpful travel assistant. Be concise and friendly.",
  // Optional: add memory
  checkpointSaver: new MemorySaver(),
});

// ── 3. Use the agent ───────────────────────────────────────────────
console.log("=== Prebuilt ReAct Agent ===\n");

const config = { configurable: { thread_id: "travel-chat-1" } };

// Query 1: Uses weather tool
console.log("--- Query 1: Weather ---");
let result = await agent.invoke(
  { messages: [new HumanMessage("What's the weather like in Mumbai?")] },
  config
);
console.log("Agent:", result.messages[result.messages.length - 1].content);

// Query 2: Uses news tool
console.log("\n--- Query 2: News ---");
result = await agent.invoke(
  { messages: [new HumanMessage("What's the latest news about AI?")] },
  config
);
console.log("Agent:", result.messages[result.messages.length - 1].content);

// Query 3: Memory test — remembers the conversation
console.log("\n--- Query 3: Memory ---");
result = await agent.invoke(
  { messages: [new HumanMessage("What city did I ask about earlier?")] },
  config
);
console.log("Agent:", result.messages[result.messages.length - 1].content);
