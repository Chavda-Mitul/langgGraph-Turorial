import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChromaClient } from "chromadb";
import { ChatGroq } from "@langchain/groq";
import { Annotation, StateGraph, START, END } from "@langchain/langgraph";
import z from "zod";
import { tavily } from "@tavily/core";
import { webSearch } from "../../09-mini-project/tools";


const UPPER_TH = 0.7
const LOWER_TH = 0.3

const model = new ChatGroq({ model: "llama-3.3-70b-versatile", temperature: 0.4, maxTokens:300 });
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

const GraphState = Annotation.Root({
    question: Annotation<string>,
    docs: Annotation<string[]>,

    good_docs: Annotation<string[]>,
    verdict: Annotation<string>,
    reason: Annotation<string>,

    strips:Annotation<string[]>,
    kept_strips:Annotation<string[]>,
    refined_context: Annotation<string>,

    web_docs: Annotation<string[]>,

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
    const docs = results.documents[0] as string[];
    return { docs };
};

// -----------------------------
// Score-based doc evaluator
// -----------------------------

const docEvalScoreSchema = z.object({
    score:z.number().describe('score between 0.0 to 1.0'),
    reason: z.string().describe('Reson for the given score'),
});

const docEvalScoreModel = model.withStructuredOutput(docEvalScoreSchema, { method: 'functionCalling' });

const docEvalScoreNode = async (state: typeof GraphState.State) => {
    const results = await Promise.all(
        state.docs.map((doc) => docEvalScoreModel.invoke(`
            You are a strict retrieval evaluator for RAG.
            Score how relevant this chunk is to the question.
            - 1.0: chunk fully/mostly answers the question
            - 0.0: chunk is irrelevant
            Be conservative with high scores.
            query: ${state.question}
            doc: ${doc}
        `))
    );

    const good_docs = results
    .map((r, i) => ({ ...r, doc: state.docs[i] })) // 👈 attach manually
    .filter(r => r.score > LOWER_TH)
    .map(r => r.doc);

    const scores = results.map(r => r.score);

      // ✅ CORRECT
    if (scores.some(s => s > UPPER_TH)) {
        return {
        good_docs,
        verdict: "CORRECT",
        reason: `At least one retrieved chunk scored > ${UPPER_TH}.`,
        };
    }

    // ❌ INCORRECT
    if (scores.length > 0 && scores.every(s => s < LOWER_TH)) {
        return {
        good_docs: [],
        verdict: "INCORRECT",
        reason: `All retrieved chunks scored < ${LOWER_TH}. No chunk was sufficient.`,
        };
    }

    // ⚖️ AMBIGUOUS
    return {
        good_docs,
        verdict: "AMBIGUOUS",
        reason: `No chunk scored > ${UPPER_TH}, but not all were < ${LOWER_TH}. Mixed relevance signals.`,
    };
};

function decomposeToSentences(text: string): string[] {
  const cleaned = text.replace(/\s+/g, " ").trim();

  const sentences = cleaned.split(/(?<=[.!?])\s+/);

  return sentences
    .map(s => s.trim())
    .filter(s => s.length > 20);
}

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

const stripFilterSchmema = z.object({
    isRelevant: z.preprocess(
    (val) => val === "true" ? true : val === "false" ? false : val,
    z.boolean().describe('wheather this line is relevent to the quary or not')
  )
});

const filterModel = model.withStructuredOutput(stripFilterSchmema);

const filterNode = async (state: typeof GraphState.State) => {
    const sourceDocs = state.verdict === 'CORRECT' ? state.good_docs : state.web_docs;
    const strips = sourceDocs.flatMap(doc => decomposeToSentences(doc));

    const results = await Promise.all(
        strips.map(sentence => filterModel.invoke(`
            You are a strict relevance filter.
            Return isRelevent ONLY if the sentence directly helps answer the question.
            question: ${state.question}
            sentence: ${sentence}
        `))
    );

    const kept_strips = strips.filter((_, i) => results[i].isRelevant);
    const refined_context = kept_strips.join("\n");

    return { strips, kept_strips, refined_context };
};

const getDocsFromWebSearch = async (state: typeof GraphState.State) => {
     const search = await tvly.search(`${state.question}`, {
        maxResults: 5, maxTokens: 500, includeImages: false,
    });
    const docs = JSON.stringify(search.results);
    console.log(docs);
    return { web_docs: search.results.map(r => r.content || r.title) };
}

const  route_after_eval = (state: typeof GraphState.State) => {
    if(state.verdict.includes('CORRECT')){
        return "filter"
    }else if(state.verdict.includes('INCORRECT')){
        return 'web_search'
    }else {
        return 'ambiguous'
    }
}


// --- WORKFLOW ---

const app = new StateGraph(GraphState)
    .addNode("retrieve", retrieveNode)
    .addNode('evaluator',docEvalScoreNode)
    .addNode('web_search',getDocsFromWebSearch)
    .addNode("filter",   filterNode)
    .addNode("generate", generateNode)
    .addEdge(START,      "retrieve")
    .addEdge("retrieve", "evaluator")
    .addConditionalEdges(
        'evaluator',
        route_after_eval,
        {
          web_search: 'web_search',
          filter:     'filter',
          ambiguous:  'web_search',
        }
    )
    .addEdge("filter",   "generate")
    .addEdge("generate", END)
    .compile();

const result = await app.invoke({ question: "what is the capital of india" });
console.log("\nAnswer:", result.answer);    

