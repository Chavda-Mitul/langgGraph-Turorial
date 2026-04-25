
import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  StateGraph,
  START,
  END,
  Annotation,
} from "@langchain/langgraph";

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0.7,
  maxTokens: 256,
}); 

const GraphState = Annotation.Root({
    perfume: Annotation<string>,
    fragneticaReview: Annotation<string>,
    redditReview: Annotation<string>,
    summary: Annotation<string>,
});

const getReviewFromReddit = async (state:typeof GraphState.State) => {
    const res = await model.invoke(`You are review colletor form the r/DesiFragranceAddicts (reddit) given perfume : ${state.perfume}`);
    return { redditReview : res.content as string };
};

const getReviewFromFragnetica = async (state : typeof GraphState.State) => {
    const res = await model.invoke(`You are review colletor form the fragnetica for a given perfume : ${state.perfume}`)
    return {fragneticaReview: res.content as string}
}

const combineReviewAndSummrise = async (state : typeof GraphState.State) => {
    const res = await model.invoke(`You are a perfume enthustiast and you have perfume knowldge you also summrise the pefume review as it will be givne to you by the fragnetica : ${state.fragneticaReview} as reddit : ${state.redditReview}`)
    return {summary : res.content as string};
}

// Fan-out: START → reddit + fragnetica (run in parallel)
// Fan-in:  both → summarizer
const workflow = new StateGraph(GraphState)
    .addNode("redditReviewer", getReviewFromReddit)
    .addNode("fragneticaReviewer", getReviewFromFragnetica)
    .addNode("summarizer", combineReviewAndSummrise)
    .addEdge(START, "redditReviewer")
    .addEdge(START, "fragneticaReviewer")
    .addEdge("redditReviewer", "summarizer")
    .addEdge("fragneticaReviewer", "summarizer")
    .addEdge("summarizer", END);

const app = workflow.compile();

const run = async () => {
    const result = await app.invoke({ perfume: "Afnan Supremacy Not Only intense" });
    console.log("Reddit Review:\n", result.redditReview);
    console.log("\nFragnetica Review:\n", result.fragneticaReview);
    console.log("\nSummary:\n", result.summary);
};

run();
