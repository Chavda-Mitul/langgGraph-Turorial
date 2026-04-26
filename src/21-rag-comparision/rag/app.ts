import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChromaClient } from "chromadb";
import { ChatGroq } from "@langchain/groq";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

const model = new ChatGroq({ model: "llama-3.3-70b-versatile", temperature: 0.4, maxTokens:300 });

const GraphState = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<string>,
    answer: Annotation<string>,
});


const pdfPaths = [
//   "src/21-rag-comparision/docs/pdf1.pdf",
//   "src/21-rag-comparision/docs/backend_system.pdf",
  "src/21-rag-comparision/docs/attention-is-all-you-need-Paper.pdf",
];

const client = new ChromaClient();

const ingestDocuments = async () => {
    const loadedDocs = await Promise.all(pdfPaths.map(p => new PDFLoader(p).load()));
    const allDocs = loadedDocs.flat();
    const splitter = new RecursiveCharacterTextSplitter({ chunkOverlap: 100, chunkSize: 512 });
    const texts = await splitter.splitDocuments(allDocs);

    try { await client.deleteCollection({ name: "rag_docs" }); } catch {}
    const col = await client.getOrCreateCollection({ name: "rag_docs" });

    await col.add({
        ids: texts.map((_, i) => `chunk-${i}`),
        documents: texts.map(t => t.pageContent),
        metadatas: texts.map(t => ({ source: t.metadata.source ?? "unknown" })),
    });
    console.log(`Stored ${texts.length} chunks in ChromaDB`);
};

// comment out after first run
// await ingestDocuments();

const collection = await client.getOrCreateCollection({ name: "rag_docs" });

// --- NODES ---

const retrieveNode = async (state: typeof GraphState.State) => {
    const results = await collection.query({
        queryTexts: [state.question],
        nResults: 5,
    });
    const context = results.documents[0].join("\n\n");
    return { context };
};

const generateNode = async (state: typeof GraphState.State) => {
    const res = await model.invoke(`
        Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        ${state.context}

        Question: ${state.question}
    `);
    return { answer: res.content as string };
};

// --- WORKFLOW ---

const app = new StateGraph(GraphState)
    .addNode("retrieve", retrieveNode)
    .addNode("generate", generateNode)
    .addEdge(START, "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", END)
    .compile();

const result = await app.invoke({ question: "What is the attention mechanism?" });
console.log("\nAnswer:", result.answer);

