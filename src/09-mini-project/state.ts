/**
 * 09-mini-project/state.ts
 * ─────────────────────────
 * State definition for the Multi-Agent Content Pipeline.
 *
 * The state flows through: Supervisor → Researcher → Writer → Reviewer → (loop or done)
 */

import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";

export const ContentPipelineState = Annotation.Root({
  // Chat messages for agent communication
  messages: Annotation<BaseMessage[]>({
    reducer: (existing, update) => [...existing, ...update],
    default: () => [],
  }),

  // User's original request
  request: Annotation<string>(),

  // Which agent should act next (decided by supervisor)
  nextAgent: Annotation<string>({
    reducer: (_, b) => b,
    default: () => "supervisor",
  }),

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
