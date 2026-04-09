# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A hands-on LangGraph tutorial in TypeScript (v1.2+) with 17 progressive modules. Uses Groq (Llama 3.3 70B) as the LLM provider. Each file under `src/` is self-contained and runnable independently.

## Commands

```bash
# Install dependencies
npm install

# Run any tutorial file (preferred — works with Node.js v24+)
npx tsx src/01-graph-basics/01-first-graph.ts

# Alternative runner (has issues on Node.js v24+)
npx ts-node --esm src/01-graph-basics/01-first-graph.ts

# Run the index (sanity check + module listing)
npm run dev

# Build TypeScript
npm run build
```

There are no tests or linting configured.

## Environment

Requires a `GROQ_API_KEY` in `.env` (loaded via `dotenv/config`). Node.js 18+.

## Architecture

**Module structure:** `src/XX-topic-name/YY-example.ts` — numbered modules from 01 (basics) to 17 (search-agent). Each file is standalone with inline comments explaining concepts.

**Key patterns used across modules:**
- **StateGraph API** (declarative): `Annotation.Root()` → `StateGraph` → `.addNode()` → `.addEdge()` → `.compile()` → `.invoke()`
- **Functional API** (imperative, module 10): `entrypoint()` + `task()` with standard async/await
- **Command routing** (module 11+): `Command({ goto, update })` for node-driven routing instead of `addConditionalEdges`

**Multi-file modules:**

- **Module 09 (mini-project)** — Supervisor-pattern multi-agent content pipeline:
  - `state.ts` → `ContentPipelineState` annotation with message history, pipeline fields, and reducers
  - `tools.ts` → Tool definitions (web search, content tools)
  - `agents.ts` → Specialist agent nodes (researcher, writer, reviewer, finalizer)
  - `graph.ts` → Graph wiring with supervisor routing via Command
  - `app.ts` → Entry point with streaming execution

- **Module 17 (search-agent)** — Research pipeline with human-in-the-loop:
  - `schema.ts` → `ProjectState` annotation (topic, rawContent, polishedContent, summary)
  - `model.ts` → Shared ChatGroq model instance
  - `nodes.ts` → Researcher, editor, summarizer nodes + conditional routing logic
  - `workflow.ts` → Graph wiring with conditional edges and interrupt-before
  - `main.ts` → Entry point with MemorySaver checkpointer and resume flow

**LLM initialization pattern** (repeated in most files):
```typescript
import "dotenv/config";
import { ChatGroq } from "@langchain/groq";
const model = new ChatGroq({ model: "llama-3.3-70b-versatile", temperature: 0.7 });
```

## TypeScript Config

- ESM modules (`"type": "module"` in package.json, `"module": "ESNext"` in tsconfig)
- `ts-node --esm` flag is required if using ts-node; prefer `tsx` instead
- Strict mode enabled

## Key Gotchas

- **ESM imports require `.js` extensions** for local files: `import { foo } from "./bar.js"` (even though the source is `.ts`). npm package imports don't need extensions.
- **Node names must not collide with state field names.** LangGraph uses the same namespace for both channels (state fields) and nodes. E.g., if state has a `summary` field, don't name a node `"summary"` — use `"summarizer"` instead.
- **Every file using ChatGroq needs `import "dotenv/config"`** at the top to load the API key from `.env`.
