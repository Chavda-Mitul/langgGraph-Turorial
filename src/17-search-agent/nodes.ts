/**
 * Node 1: Research Logic
 */

import { model } from './model.js';
import { State } from './schema.js'

export const researchNode = async (state:State) => {
    console.log(`[LOG] Researching topic: ${state.topic}...`);

    const response = await model.invoke([
        {
            role:"system",
            content: "You are a research assistant. Provide 3 short, factual sentences about the user topic."
        },
        {
            role: "user",
            content: state.topic 
        }
    ])

    return {
        rawContent: (response.content as string) || ""
    }
}

/**
 * Node 2: Editing Logic
 */

export const editingNode = async (state:State) => {
    console.log("[LOG] Polishing content...");

    const polished = await model.invoke([
        {
            role: 'system',
            content: 'you are a reasearch paper editor, you will get the raw research you have to convert that for public reading'
        },
        {
            role: "user",
            content: state.rawContent
        }
    ])

    return {
        polishedContent : (polished.content as string) || ""
    };
}

/**
 * Node 3: Summary 
 */

export const summaryNode = async (state:State) => {
    console.log("[LOG] Summary content...");

    const summaryed = await model.invoke([
        {
            role:'system',
            content: 'You are a expert at summrising the content, that does not loose the detail but make content shorter.'
        },
        {
            role:"user",
            content: state.polishedContent   
        }
    ])

    // We take the polished content and shorten it
    return {
        summary: (summaryed.content as string) || ""
    };
}

/**
 * This function is the "Brain" of the graph.
 * It doesn't change data; it just directs traffic.
 */

export const checkContentLength = (state: State) => {
    console.log("[LOG] Checking content length...");

    // If the researcher found less than 10 characters, it's a fail.    

    if(state.rawContent.length < 10){
        console.log("[LOG] Content too short! Ending early.");
        return "too_short";
    }
    else {
        console.log("[LOG] Content looks good. Moving to Editor.");
        return "is_good";
    }
}