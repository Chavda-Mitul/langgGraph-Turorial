import { StateGraph, START, END, Annotation } from "@langchain/langgraph";

/**
 * @module State
 * This defines the shape of our application's memory.
 */
export const ProjectState = Annotation.Root({
  topic: Annotation<string>,
  rawContent: Annotation<string>,
  polishedContent: Annotation<string>,
  summary: Annotation<string>
});

// We extract the Type for use in our functions
export type State = typeof ProjectState.State;