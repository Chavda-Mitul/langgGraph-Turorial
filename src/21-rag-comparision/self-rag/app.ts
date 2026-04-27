import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChromaClient } from "chromadb";
import { ChatGroq } from "@langchain/groq";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import z from "zod";
import { tavily } from "@tavily/core";

const model = new ChatGroq({ model: "llama-3.3-70b-versatile", temperature: 0.4, maxTokens:100 });
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

