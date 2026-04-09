/**
 * 09-mini-project/state.ts
 * ─────────────────────────
 * State definition for the Multi-Agent Content Pipeline.
 *
 * Uses modern LangGraph patterns:
 * - MessagesAnnotation.spec for message handling
 * - Custom fields with reducers for tracking pipeline progress
 * - Command routing (no nextAgent field needed — Command handles routing)
 */

import { Annotation, MessagesAnnotation } from "@langchain/langgraph";

export const ContentPipelineState = Annotation.Root({
  // Inherit messages with built-in reducer from MessagesAnnotation
  ...MessagesAnnotation.spec,

  // User's original request
  request: Annotation<string>(),

  // Research data gathered
  research: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),

  // Draft content written
  draft: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),

  // Review feedback
  feedback: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),

  // Final polished content
  finalContent: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "",
  }),

  // Pipeline status
  status: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "started",
  }),

  // Iteration count (for revision loop)
  iteration: Annotation<number>({
    reducer: (a, b) => a + b,
    default: () => 0,
  }),

  // Activity log
  log: Annotation<string[]>({
    reducer: (existing, update) => [...existing, ...update],
    default: () => [],
  }),
});

export type ContentState = typeof ContentPipelineState.State;
