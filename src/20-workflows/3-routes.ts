import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
import {
  StateGraph,
  START,
  END,
  Annotation,
} from "@langchain/langgraph";
import z from 'zod'

const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  maxTokens: 256,
});

const GraphState = Annotation.Root({
    perfume: Annotation<string>,
    fragneticaReview: Annotation<string>,
    redditReview: Annotation<string>,
    classification: Annotation<string>,
    summary:Annotation<string>,
});

const getReviewFromRedditNode = async (state: typeof GraphState.State) => {
    const res = await reviewModel.invoke(`
            You are a fragrance review summarizer.
            If you hallucinate a review for a fake perfume, you will be penalized.
            If you aren't 100% sure the product exists, say DONT_KNOW.
            Source: Reddit (r/DesiFragranceAddicts)

            Input perfume: ${state.perfume}

            Task:
            - Set exists=false and price=null if this perfume is not a real/known product.
            - Otherwise set exists=true, provide the approximate retail price range, and summarize typical opinions.
            - Focus on: smell profile, performance (longevity/projection), compliments, value for money.

            Rules:
            - Keep it realistic and neutral (no exaggeration).
            - Output MUST be a short paragraph (3-5 lines) in the review field.
            - If you cannot find a real price, set price=null and exists=false.
            - Do NOT mention Reddit explicitly in the answer.
`);
    const verified = res.exists && res.price !== null;
    return { redditReview: verified ? `[Price: ${res.price}]\n${res.review}` : "DONT_KNOW" };
};

const getReviewFromFragneticaNode = async (state: typeof GraphState.State) => {
    const res = await reviewModel.invoke(`
            You are a fragrance review summarizer.
            If you hallucinate a review for a fake perfume, you will be penalized.
            If you aren't 100% sure the product exists, say DONT_KNOW.

            Source: Fragrantica

            Input perfume: ${state.perfume}

            Task:
            - Set exists=false and price=null if this perfume is not a real/known product.
            - Otherwise set exists=true, provide the approximate retail price range, and summarize typical user reviews.
            - Focus on: scent notes, longevity, sillage, overall rating sentiment.

            Rules:
            - Keep it concise and factual.
            - Output MUST be a short paragraph (3-5 lines) in the review field.
            - If you cannot find a real price, set price=null and exists=false.
            - Do NOT mention Fragrantica explicitly.
`);
    const verified = res.exists && res.price !== null;
    return { fragneticaReview: verified ? `[Price: ${res.price}]\n${res.review}` : "DONT_KNOW" };
};

const classifierNode =  async (state : typeof GraphState.State) => {
    const res = await model.invoke(`
            You are a strict perfume category classifier.

            Input: A perfume name : ${state.perfume}

            Task:
            Identify the brand from the perfume name and classify it into ONE category.

            Categories:
            - MIDDLE_EASTERN_PERFUME → Afnan, Rasasi, Ahmed Al Maghribi, French Avenue, Lattafa, Armaf, Ajmal, etc.
            - DESIGNER_PERFUME → Dior, Chanel, Zara, Tom Ford, Xerjoff, YSL, Gucci, Prada, etc.

            Rules:
            - Output ONLY one of these exact values:
            MIDDLE_EASTERN_PERFUME
            DESIGNER_PERFUME
            DONT_KNOW

            - Do NOT explain.
            - Do NOT guess if unsure.
            - If brand is unclear or not in known lists → return DONT_KNOW.
            - Be conservative in classification (prefer DONT_KNOW over wrong classification).
 `);

    return {classification: res.content as string};
}

const generatePerfumeReportNode = async (state: typeof GraphState.State) => {
    const relevantReview = state.classification.includes("MIDDLE_EASTERN_PERFUME")
        ? state.redditReview
        : state.fragneticaReview;

    if (!relevantReview || relevantReview === "DONT_KNOW") {
        return { summary: "No information available for this perfume. Please verify the name." };
    }

    const res = await model.invoke(`
            You are a perfume advisor.
            First, verify if this perfume exists.
            If it does not exist or you have zero specific data about it, you MUST return exactly 'DONT_KNOW'.
            Do not describe the brand in general. 
            Do not guess the scent profile based on the name of the perfume.
            Input:
            - Perfume name: ${state.perfume}
            - Category: ${state.classification}
            - Reddit review: ${state.redditReview || "NOT_AVAILABLE"}
            - Fragrantica review: ${state.fragneticaReview || "NOT_AVAILABLE"}

            Context:
            - If category = MIDDLE_EASTERN_PERFUME → Reddit review is the source.
            - If category = DESIGNER_PERFUME → Fragrantica review is the source.

            Task:
            1. Clean and correct the perfume name (fix spelling, proper capitalization).
            2. Use ONLY the available review (ignore NOT_AVAILABLE).
            3. Summarize:
            - Scent profile
            - Performance (longevity/projection)
            - Overall sentiment

            Decision:
            - If review is "DONT_KNOW" or NOT_AVAILABLE → return:
            "No information available for this perfume. Please verify the name."

            - Otherwise give verdict:
            BUY → mostly positive
            SKIP → mostly negative
            CONSIDER → mixed

            Output format (STRICT):
            Perfume: <clean name>

            Summary:
            <3-5 lines>

            Verdict: <BUY | SKIP | CONSIDER>

            Rules:
            - Do NOT hallucinate.
            - Use only given data.
            - Keep concise.
`);

    return { summary: res.content as string };
};

const routeNode = (state: typeof GraphState.State) => {
    if (state.classification.includes("MIDDLE_EASTERN_PERFUME")) {
        return "review_reddit";
    }else if(state.classification.includes("DESIGNER_PERFUME")){
        return "review_fragnetica"
    }else {
        return "dont_know";
    }
};

const ReviewSchema = z.object({
  exists: z.boolean().describe("Whether this specific perfume is a real product"),
  price: z.string().nullable().describe("Approximate retail price range (e.g. '$30-$50'). Set to null if you cannot find a real price."),
  review: z.string().describe("The summary of the reviews")
});

const reviewModel = model.withStructuredOutput(ReviewSchema);



const workflow = new StateGraph(GraphState)
                     .addNode('classifier', classifierNode)
                     .addNode('review_fragnetica', getReviewFromFragneticaNode)
                     .addNode('review_reddit', getReviewFromRedditNode)
                     .addNode('summarizer', generatePerfumeReportNode)
                     .addEdge(START, 'classifier')
                     .addConditionalEdges(
                        'classifier',
                        routeNode,
                        {
                            review_fragnetica: 'review_fragnetica',
                            review_reddit: 'review_reddit',
                            dont_know: 'summarizer',
                        }
                     )
                     .addEdge('review_fragnetica', 'summarizer')
                     .addEdge('review_reddit', 'summarizer')
                     .addEdge('summarizer', END);

const app = workflow.compile();
const run = async () => {
    const res = await app.invoke({ perfume: 'rasasi winter' });
    console.log(res.classification);
    console.log(res.fragneticaReview);
    console.log(res.redditReview);
    console.log(res.summary);
};

run();