
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
    topic: Annotation<String>,
    research: Annotation<String>,
    critique: Annotation<String>
});

const doResearh = async (state: typeof GraphState.State) => {
    const res = await model.invoke(`Generate a business idea for: ${state.topic}`);
    return {research:res.content as string};
}   

const critiqueIdea = async (state: typeof GraphState.State) => {
    const res = await model.invoke(`Give a short critique of this idea: ${state.research}`);
    return {critique:res.content as string};
};

//build node
const workflow = new StateGraph(GraphState)
                     .addNode('researcher',doResearh)
                     .addNode('critiquer',critiqueIdea)
                     .addEdge(START,'researcher')
                     .addEdge('researcher','critiquer')
                     .addEdge('critiquer',END)

const app = workflow.compile();

const run = async ()=> {
    const result = await app.invoke({topic:'Perfume decanting in minaturs bottles'});
    console.log(result);
} 
run();
  