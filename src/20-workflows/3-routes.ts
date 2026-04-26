import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { tavily } from "@tavily/core";
import z from "zod";

// --- SETUP ---

const model = new ChatGroq({ model: "llama-3.3-70b-versatile", temperature: 0, maxTokens: 256 });
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

const GraphState = Annotation.Root({
    perfume:          Annotation<string>,
    perfumePrice:     Annotation<string>,
    classification:   Annotation<string>,
    redditReview:     Annotation<string>,
    fragneticaReview: Annotation<string>,
    summary:          Annotation<string>,
});

// --- SCHEMAS ---

const ClassifierSchema = z.object({
    classification: z.enum(["MIDDLE_EASTERN_PERFUME", "DESIGNER_PERFUME", "DONT_KNOW"]),
    perfumePrice:   z.string().nullable().describe("Price range from web search e.g. '₹2000-₹4500', null if not found"),
});

const ReviewSchema = z.object({
    exists: z.boolean().describe("Whether this perfume is a real known product"),
    review: z.string().describe("Short review paragraph (3-5 lines)"),
});

const classifierModel = model.withStructuredOutput(ClassifierSchema);
const reviewModel     = model.withStructuredOutput(ReviewSchema);

// --- NODES ---

const classifierNode = async (state: typeof GraphState.State) => {
    const search = await tvly.search(`Price of ${state.perfume} perfume in India`, {
        maxResults: 3, maxTokens: 100, includeImages: false,
    });

    const res = await classifierModel.invoke(`
        Perfume: ${state.perfume}
        Web search results: ${JSON.stringify(search.results)}

        Classify the brand:
        - MIDDLE_EASTERN_PERFUME → Afnan, Rasasi, Lattafa, Armaf, Ajmal, Ahmed Al Maghribi, etc.
        - DESIGNER_PERFUME → Dior, Chanel, Tom Ford, YSL, Gucci, Prada, etc.
        - DONT_KNOW → brand unclear or unknown
        Extract perfumePrice from web results, or null if not found.
    `);

    return { classification: res.classification, perfumePrice: res.perfumePrice ?? "Not found" };
};

const redditReviewNode = async (state: typeof GraphState.State) => {
    const res = await reviewModel.invoke(`
        Source: Reddit (r/DesiFragranceAddicts)
        Perfume: ${state.perfume}

        Set exists=false if this is not a real product.
        Otherwise summarize: scent profile, longevity, projection, value for money.
        Do NOT mention Reddit.
    `);
    return { redditReview: res.exists ? res.review : "DONT_KNOW" };
};

const fragneticaReviewNode = async (state: typeof GraphState.State) => {
    const res = await reviewModel.invoke(`
        Source: Fragrantica
        Perfume: ${state.perfume}

        Set exists=false if this is not a real product.
        Otherwise summarize: scent notes, longevity, sillage, overall sentiment.
        Do NOT mention Fragrantica.
    `);
    return { fragneticaReview: res.exists ? res.review : "DONT_KNOW" };
};

const reportNode = async (state: typeof GraphState.State) => {
    const review = state.classification === "MIDDLE_EASTERN_PERFUME"
        ? state.redditReview
        : state.fragneticaReview;

    if (!review || review === "DONT_KNOW") {
        return { summary: "No information available for this perfume. Please verify the name." };
    }

    const res = await model.invoke(`
        You are a perfume advisor. Write a concise report using ONLY the data below.

        Perfume: ${state.perfume}
        Price: ${state.perfumePrice}
        Review: ${review}

        Format:
        Perfume: <corrected name>
        Price: <price>
        Summary: <3-5 lines on scent, performance, sentiment>
        Verdict: BUY | SKIP | CONSIDER
    `);
    return { summary: res.content as string };
};

const routeNode = (state: typeof GraphState.State) => {
    if (state.classification === "MIDDLE_EASTERN_PERFUME") return "review_reddit";
    if (state.classification === "DESIGNER_PERFUME")       return "review_fragnetica";
    return "dont_know";
};

// --- GRAPH ---

const app = new StateGraph(GraphState)
    .addNode("classifier",        classifierNode)
    .addNode("review_reddit",     redditReviewNode)
    .addNode("review_fragnetica", fragneticaReviewNode)
    .addNode("summarizer",        reportNode)
    .addEdge(START, "classifier")
    .addConditionalEdges("classifier", routeNode, {
        review_reddit:     "review_reddit",
        review_fragnetica: "review_fragnetica",
        dont_know:         "summarizer",
    })
    .addEdge("review_reddit",     "summarizer")
    .addEdge("review_fragnetica", "summarizer")
    .addEdge("summarizer", END)
    .compile();

// --- RUN ---

const res = await app.invoke({ perfume: "Rasasi MaxMax" });
console.log("Classification:", res.classification);
console.log("Price:",          res.perfumePrice);
console.log("Summary:\n",      res.summary);
