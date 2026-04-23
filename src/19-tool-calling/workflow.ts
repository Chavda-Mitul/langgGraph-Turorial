import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { model } from "./model";
import { tools } from "./tools";

// Node 
const callModel = async (state: typeof MessagesAnnotation.State) => {
    const response = await model.invoke(state.messages);
    return {messages:[response]};
}

const toolNode = new ToolNode(tools);

// Routing Logic: Should we continue to tools or end?

const shouldContinue = (state:typeof MessagesAnnotation.State) => {
    const {messages} = state;
    const lastMessage = messages[messages.length - 1];

    // If the LLM made a tool call, go to the "tools" node
    if (lastMessage.additional_kwargs.tool_calls) {
    return "tools";
  }

    // Otherwise, we are done!
    return END;
}

// Assembly
const workflow = new StateGraph(MessagesAnnotation)
                     .addNode('agent',callModel)
                     .addNode('tools',toolNode)
                     .addEdge(START,'agent')
                     .addConditionalEdges('agent',shouldContinue)
                     .addEdge('tools','agent')

export const app = workflow.compile();
app.invoke({ messages: [{ role: "user", content: "What is the price of MSFT?" }] })