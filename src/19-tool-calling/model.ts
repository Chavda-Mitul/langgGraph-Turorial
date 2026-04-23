import { ChatGroq } from "@langchain/groq";
import {tools} from './tools'

export const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama-3.3-70b-versatile",
}).bindTools(tools); // 👈 This "binds" the tools to the LLM