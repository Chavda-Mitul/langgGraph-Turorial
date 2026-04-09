import { END, START, StateGraph } from '@langchain/langgraph';
import {checkContentLength, editingNode,researchNode,summaryNode} from './nodes.js';
import {ProjectState} from './schema.js';
import { MemorySaver } from "@langchain/langgraph";


export const workflow = new StateGraph(ProjectState)
                // Step A: Register the Nodes
                 .addNode("researcher", researchNode)
                 .addNode('editor',editingNode)
                 .addNode('summarizer',summaryNode)

                // Step B: Connect the Nodes with Edges
                .addEdge(START, "researcher")      // Start -> Researcher
                /**
                 * 2. NEW: Conditional Logic
                 * We tell the graph: "From researcher, run checkContentLength."
                 * Then we provide a Map:
                 * If it returns "is_good" -> Go to "editor"
                 * If it returns "too_short" -> Go to END
                 */
                .addConditionalEdges('researcher',checkContentLength,{
                    'is_good' : 'editor',
                    'too_short' : END
                })
                .addEdge('editor','summarizer')       // Editor -> Summarizer
                .addEdge("summarizer", END);          // Summarizer -> Finish


