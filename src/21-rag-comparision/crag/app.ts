import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChromaClient } from "chromadb";
import { ChatGroq } from "@langchain/groq";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import { tavily } from "@tavily/core";
import z from "zod";

// --- SETUP ---

const UPPER_TH = 0.7;
const LOWER_TH = 0.3;

const model = new ChatGroq({ model: "llama-3.3-70b-versatile", temperature: 0, maxTokens: 512 });
const tvly  = tavily({ apiKey: process.env.TAVILY_API_KEY });

const GraphState = Annotation.Root({
    question:        Annotation<string>,
    docs:            Annotation<string[]>,
    good_docs:       Annotation<string[]>,
    verdict:         Annotation<string>,
    web_query:       Annotation<string>,
    web_docs:        Annotation<string[]>,
    strips:          Annotation<string[]>,
    kept_strips:     Annotation<string[]>,
    refined_context: Annotation<string>,
    answer:          Annotation<string>,
});

// --- INGEST ---

const client = new ChromaClient();

const ingestDocuments = async () => {
    const pdfPaths = [
        "src/21-rag-comparision/docs/attention-is-all-you-need-Paper.pdf",
    ];
    const allDocs = (await Promise.all(pdfPaths.map(p => new PDFLoader(p).load()))).flat();
    const texts   = await new RecursiveCharacterTextSplitter({ chunkSize: 512, chunkOverlap: 100 }).splitDocuments(allDocs);

    try { await client.deleteCollection({ name: "rag_docs" }); } catch {}
    const col = await client.getOrCreateCollection({ name: "rag_docs" });
    await col.add({
        ids:       texts.map((_, i) => `chunk-${i}`),
        documents: texts.map(t => t.pageContent),
        metadatas: texts.map(t => ({ source: t.metadata.source ?? "unknown" })),
    });
    console.log(`Stored ${texts.length} chunks in ChromaDB`);
};

// comment out after first run
// await ingestDocuments();

const collection = await client.getOrCreateCollection({ name: "rag_docs" });

// --- SCHEMAS ---

const evalSchema = z.object({
    score:  z.number().describe("Relevance score 0.0 to 1.0"),
    reason: z.string().describe("Short reason for the score"),
});

const filterSchema = z.object({
    isRelevant: z.preprocess(
        val => val === "true" ? true : val === "false" ? false : val,
        z.boolean().describe("true if sentence directly helps answer the question")
    ),
});

const querySchema = z.object({
    query: z.string().describe("Short web search query (6-14 keywords)"),
});

const evalModel   = model.withStructuredOutput(evalSchema,   { method: "functionCalling" });
const filterModel = model.withStructuredOutput(filterSchema);
const queryModel  = model.withStructuredOutput(querySchema);

// --- HELPERS ---

function toSentences(text: string): string[] {
    return text
        .replace(/\s+/g, " ")
        .trim()
        .split(/(?<=[.!?])\s+/)
        .map(s => s.trim())
        .filter(s => s.length > 20);
}

// --- NODES ---

const retrieveNode = async (state: typeof GraphState.State) => {
    const results = await collection.query({ queryTexts: [state.question], nResults: 3 });
    return { docs: results.documents[0] as string[] };
};

const evaluatorNode = async (state: typeof GraphState.State) => {
    const results = await Promise.all(
        state.docs.map(doc => evalModel.invoke(`
            You are a strict RAG retrieval evaluator.
            Score how relevant this chunk is to the question.
            1.0 = fully answers it. 0.0 = completely irrelevant. Be conservative.
            question: ${state.question}
            chunk: ${doc}
        `))
    );

    const scores    = results.map(r => r.score);
    const good_docs = state.docs.filter((_, i) => scores[i] > LOWER_TH);

    console.log("Scores:", scores.map((s, i) => `doc${i}=${s.toFixed(2)}`).join(" | "));

    if (scores.some(s => s > UPPER_TH))              return { good_docs, verdict: "CORRECT" };
    if (scores.every(s => s < LOWER_TH))             return { good_docs: [], verdict: "INCORRECT" };
    return { good_docs, verdict: "AMBIGUOUS" };
};

const rewriteQueryNode = async (state: typeof GraphState.State) => {
    const res = await queryModel.invoke(`
        Rewrite this question into a short web search query (6-14 keywords only).
        Do NOT answer the question.
        question: ${state.question}
    `);
    console.log("Web query:", res.query);
    return { web_query: res.query };
};

const webSearchNode = async (state: typeof GraphState.State) => {
    const search = await tvly.search(state.web_query, {
        maxResults: 3, maxTokens: 300, includeImages: false,
    });
    const fetched = search.results.map(r => r.content || r.title).filter(Boolean) as string[];
    console.log(`Web search returned ${fetched.length} docs`);

    const web_docs = state.verdict === "AMBIGUOUS"
        ? [...state.good_docs, ...fetched]
        : fetched;

    return { web_docs };
};

const filterNode = async (state: typeof GraphState.State) => {
    const sourceDocs = state.verdict === "CORRECT" ? state.good_docs : (state.web_docs ?? []);
    const strips     = sourceDocs.flatMap(toSentences);

    const results = await Promise.all(
        strips.map(sentence => filterModel.invoke(`
            You are a strict relevance filter.
            Return isRelevant=true ONLY if the sentence directly helps answer the question.
            question: ${state.question}
            sentence: ${sentence}
        `))
    );

    const kept_strips     = strips.filter((_, i) => results[i].isRelevant);
    const refined_context = kept_strips.length > 0 ? kept_strips.join("\n") : strips.join("\n");

    console.log(`Filter: kept ${kept_strips.length}/${strips.length} strips`);
    return { strips, kept_strips, refined_context };
};

const generateNode = async (state: typeof GraphState.State) => {
    const res = await model.invoke(`
        Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        ${state.refined_context}

        Question: ${state.question}
    `);
    return { answer: res.content as string };
};

const routeAfterEval = (state: typeof GraphState.State) => {
    if (state.verdict === "CORRECT") return "filter";
    return "rewrite_query";
};

// --- GRAPH ---

const app = new StateGraph(GraphState)
    .addNode("retrieve",      retrieveNode)
    .addNode("evaluator",     evaluatorNode)
    .addNode("rewrite_query", rewriteQueryNode)
    .addNode("web_search",    webSearchNode)
    .addNode("filter",        filterNode)
    .addNode("generate",      generateNode)
    .addEdge(START,           "retrieve")
    .addEdge("retrieve",      "evaluator")
    .addConditionalEdges("evaluator", routeAfterEval, {
        filter:       "filter",
        rewrite_query: "rewrite_query",
    })
    .addEdge("rewrite_query", "web_search")
    .addEdge("web_search",    "filter")
    .addEdge("filter",        "generate")
    .addEdge("generate",      END)
    .compile();

// --- RUN ---

const result = await app.invoke({ question: "Explain attention mechanism" });
console.log("\nVerdict:", result.verdict);
console.log("\nAnswer:", result.answer);
