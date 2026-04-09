# LangGraph Tutorial — Complete Guide (TypeScript)

A comprehensive, hands-on tutorial for learning **LangGraph** — the framework for building stateful, multi-actor AI applications with LLMs using directed graphs.

> **LangGraph Version:** v1.2+ (latest APIs as of 2025)
> **Language:** TypeScript / Node.js
> **LLM Provider:** Groq (Llama 3.3 70B) — easily swappable

---

## Table of Contents

1. [What is LangGraph?](#what-is-langgraph)
2. [Core Mental Model](#core-mental-model)
3. [Tutorial Structure](#tutorial-structure)
4. [Setup](#setup)
5. [Module Guide with Theory](#module-guide-with-theory)
6. [Two APIs: StateGraph vs Functional](#two-apis-stategraph-vs-functional)
7. [Production Considerations](#production-considerations)
8. [Learning Path](#learning-path)

---

## What is LangGraph?

**LangGraph** is a low-level orchestration framework for building AI agents and workflows. Built by the LangChain team, it models your application as a **directed graph** where:

- **Nodes** = Units of work (LLM calls, tool execution, logic)
- **Edges** = Connections that define flow (static or conditional)
- **State** = Shared data that flows through the graph

### Why Graphs for AI?

Traditional software uses linear control flow: `function A → function B → function C`. But AI agents need something different:

```
Think → Act → Observe → Think again → Maybe act differently → ...
```

This is a **cycle** — and cycles need graphs, not chains. LangGraph makes this pattern first-class:

```
              ┌─────────────────────────┐
              ↓                         │
START → Agent (LLM) → Has tool calls? ──┘ (yes → tools → back to agent)
              │
              └── No → END
```

### LangGraph vs LangChain

| Feature | LangChain | LangGraph |
|---|---|---|
| Focus | Chains (linear pipelines) | Graphs (cycles, branching) |
| State | Passed through chain | Persistent, shared across nodes |
| Control Flow | Sequential | Conditional, parallel, cyclic |
| Memory | Add-on | Built-in (checkpointers) |
| Human-in-Loop | Manual | Native (interrupt/resume) |
| Best For | Simple RAG, prompts | Agents, multi-step workflows |

**Use LangChain** when you have a simple pipeline (prompt → LLM → output).
**Use LangGraph** when you need cycles, state management, multi-agent coordination, or human-in-the-loop.

---

## Core Mental Model

### The 5 Pillars of LangGraph

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. STATE          Your data (what flows through the graph)     │
│     (Annotation)   Defined once, shared by all nodes            │
│                                                                 │
│  2. NODES          Units of work (functions)                    │
│     (Functions)    Read state → Do something → Return updates   │
│                                                                 │
│  3. EDGES          Connections between nodes                    │
│     (Flow)         Static (always) or Conditional (if/else)     │
│                                                                 │
│  4. PERSISTENCE    Memory that survives between invocations     │
│     (Checkpoints)  MemorySaver (dev) or DB-backed (production)  │
│                                                                 │
│  5. HUMAN CONTROL  Pause, inspect, approve, modify              │
│     (Interrupt)    interrupt() + Command({ resume })            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### How State Works (Reducers)

State is the most important concept. Every node reads from and writes to a shared state object.

```
Without Reducer:    Last write wins (overwrite)
                    Node A sets name="Alice"
                    Node B sets name="Bob"
                    Result: name="Bob"

With Reducer:       You control the merge
                    reducer: (old, new) => [...old, ...new]
                    Node A adds ["step1"]
                    Node B adds ["step2"]
                    Result: ["step1", "step2"]
```

### The Agent Loop (ReAct Pattern)

Most AI agents follow this pattern:

```
1. LLM THINKS     → "I should search for information"
2. TOOL EXECUTES  → search("LangGraph") → results
3. LLM OBSERVES   → "Based on the results, I now know..."
4. LLM DECIDES    → Continue (more tools) or Finish (respond)
```

In LangGraph, this is a simple loop:

```
START → agent ──→ shouldContinue? ──→ tools ──→ agent (loop!)
                        │
                        └── No tools needed → END
```

---

## Tutorial Structure

```
src/
├── 01-graph-basics/           ← Start here!
│   ├── 01-first-graph.ts          Build your first graph
│   ├── 02-multi-node-graph.ts     Chain multiple nodes
│   └── 03-parallel-nodes.ts       Fan-out / fan-in
│
├── 02-state-management/       ← Understand state
│   ├── 01-annotations.ts         Define state schemas
│   └── 02-reducers.ts            Control how state merges
│
├── 03-conditional-edges/      ← Dynamic flow control
│   ├── 01-basic-routing.ts       Route by state values
│   └── 02-loops.ts               Create cycles (agent loops)
│
├── 04-tool-integration/       ← Give agents tools
│   ├── 01-tool-node.ts           ToolNode + tool calling
│   └── 02-custom-tool-graph.ts   Domain-specific tools
│
├── 05-memory-checkpoints/     ← Persistence
│   ├── 01-memory-saver.ts        Thread-based memory
│   └── 02-state-inspection.ts    Time travel & state inspection
│
├── 06-human-in-the-loop/      ← Human oversight
│   ├── 01-interrupt-approve.ts   Pause & approve workflows
│   └── 02-tool-approval.ts       Approve tool calls
│
├── 07-streaming/              ← Real-time output
│   ├── 01-stream-modes.ts        values vs updates modes
│   └── 02-token-streaming.ts     Token-by-token streaming
│
├── 08-prebuilt-agents/        ← High-level API
│   ├── 01-react-agent.ts         createReactAgent (one-liner)
│   └── 02-agent-with-state.ts    Custom state + dynamic prompt
│
├── 09-mini-project/           ← Full project
│   ├── state.ts                  Multi-agent content pipeline
│   ├── tools.ts                  with Command routing,
│   ├── agents.ts                 InMemoryStore, streaming,
│   ├── graph.ts                  and supervisor pattern
│   └── app.ts
│
├── 10-functional-api/         ← Alternative API
│   ├── 01-entrypoint-task.ts     entrypoint + task basics
│   ├── 02-memory-functional.ts   Memory without StateGraph
│   └── 03-branching-functional.ts Branching, parallel, loops
│
├── 11-command-routing/        ← Modern routing
│   ├── 01-command-basics.ts      Command({ goto, update })
│   └── 02-agent-handoffs.ts     Multi-agent handoffs
│
├── 12-long-term-memory/       ← Cross-thread persistence
│   └── 01-in-memory-store.ts     InMemoryStore for user prefs
│
├── 13-subgraphs/              ← Composition (NEW)
│   ├── 01-nested-graphs.ts       Subgraphs with state mapping
│   └── 02-shared-state-subgraphs.ts Shared state composition
│
├── 14-error-handling/         ← Resilience (NEW)
│   ├── 01-retry-and-fallback.ts  Retry + fallback patterns
│   └── 02-node-retry-decorator.ts Reusable retry wrapper
│
├── 15-message-trimming/       ← Context management (NEW)
│   └── 01-trim-messages.ts       trimMessages + summarization
│
└── 16-visualization/          ← Debugging (NEW)
    └── 01-graph-visualization.ts  Mermaid diagrams
```

---

## Setup

### Prerequisites
- Node.js 18+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd langgraph-tutorial

# Install dependencies
npm install

# Set up environment
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Running Examples

Each file is self-contained. Run any example directly:

```bash
# Run a specific tutorial
npx ts-node --esm src/01-graph-basics/01-first-graph.ts

# Run the mini-project
npx ts-node --esm src/09-mini-project/app.ts
```

---

## Module Guide with Theory

### Module 1: Graph Basics

**Theory:** Every LangGraph application is built from three primitives:

| Primitive | What it is | Analogy |
|---|---|---|
| `StateGraph` | Container for your graph | A blueprint/class |
| Node | A function that processes state | A step in a flowchart |
| Edge | Connection between nodes | An arrow in a flowchart |

**The lifecycle:**
1. **Define** state (what data flows through)
2. **Add** nodes (functions that process state)
3. **Connect** nodes with edges (define the flow)
4. **Compile** the graph (lock it, make it executable)
5. **Invoke** the compiled graph (run it!)

**Key insight:** Nodes are just functions `(state) => stateUpdate`. They receive the full state and return ONLY the fields they want to change. LangGraph handles merging.

### Module 2: State Management

**Theory: Annotations**

State in LangGraph is defined using `Annotation.Root()` — think of it as a runtime-enforced TypeScript interface:

```typescript
// This is like defining an interface, but with runtime behavior
const MyState = Annotation.Root({
  name: Annotation<string>(),           // Simple field
  items: Annotation<string[]>({         // Field with reducer
    reducer: (old, new) => [...old, ...new],
    default: () => [],
  }),
});
```

**Theory: Reducers — the secret sauce**

Without a reducer, fields use "last write wins" — if two nodes write the same field, the last one wins. Reducers give you control:

```
No reducer:     "Alice" then "Bob"      → "Bob"
Append reducer: ["a"] then ["b"]        → ["a", "b"]
Max reducer:    75 then 92 then 88      → 92
Object merge:   {a:1} then {b:2}       → {a:1, b:2}
```

`MessagesAnnotation` is a prebuilt annotation with an append reducer for messages — it's what makes chat history work automatically.

### Module 3: Conditional Edges

**Theory:** Static edges (`addEdge`) always go to the same destination. Conditional edges (`addConditionalEdges`) call a **router function** that examines state and returns a node name:

```
Static:       A ──────→ B (always)
Conditional:  A ──?──→ B (if condition)
                  └──→ C (else)
```

**Loops** are just conditional edges that point BACKWARD. This is the foundation of agentic behavior — the ability to "try again" until satisfied.

### Module 4: Tool Integration

**Theory: The ReAct Pattern**

ReAct (Reasoning + Acting) is the most common agent pattern:

1. LLM receives messages + tool descriptions
2. LLM decides: call a tool OR respond directly
3. If tool call → execute tool → feed result back to LLM
4. Repeat until LLM responds without tool calls

LangGraph provides `ToolNode` — a prebuilt node that automatically executes tool calls from AI messages.

### Module 5: Memory & Checkpoints

**Theory: Short-term vs Long-term Memory**

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  SHORT-TERM (MemorySaver / Checkpointer)               │
│  ├─ Scope: Within a single conversation (thread)       │
│  ├─ Data: Messages, state snapshots                    │
│  ├─ Lifecycle: Lives as long as the thread exists       │
│  └─ Use: Chat history, conversation continuity          │
│                                                        │
│  LONG-TERM (InMemoryStore / BaseStore)                  │
│  ├─ Scope: Across ALL conversations (all threads)       │
│  ├─ Data: User preferences, learned facts              │
│  ├─ Lifecycle: Persistent, survives thread deletion     │
│  └─ Use: User profiles, cross-session learning          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**Thread IDs** are like session IDs — each `thread_id` gets its own isolated conversation with its own state history.

### Module 6: Human-in-the-Loop

**Theory:** AI agents shouldn't always act autonomously. `interrupt()` pauses the graph and returns control to the human:

```
Graph running → interrupt("Review this plan") → Graph FROZEN
                                                     │
Human reviews → Command({ resume: "approved" }) → Graph RESUMES
```

The checkpointer saves the FULL graph state at the interrupt point, so resuming continues exactly where it stopped.

### Module 7: Streaming

**Theory:** Two levels of streaming:

```
Level 1: NODE-LEVEL streaming (graph.stream())
  "updates" mode → See what each node changed
  "values" mode  → See full state after each node

Level 2: TOKEN-LEVEL streaming (graph.streamEvents())
  See individual tokens as the LLM generates them
  Events: on_llm_start, on_chat_model_stream, on_llm_end
```

Node-level streaming is great for progress indicators. Token-level streaming creates the ChatGPT-like "typing" experience.

### Module 8: Prebuilt Agents

**Theory:** `createReactAgent()` is a one-liner that builds the full ReAct agent graph for you:

```typescript
const agent = createReactAgent({
  llm: model,
  tools: [tool1, tool2],
  prompt: "You are a helpful assistant.",  // string or function
  checkpointSaver: new MemorySaver(),
});
```

**When to use prebuilt vs custom:**
- **Prebuilt:** Standard agent behavior, quick prototyping
- **Custom StateGraph:** Multi-agent, custom routing, special state, subgraphs

### Module 9: Mini Project

**Theory: Supervisor Pattern**

The supervisor pattern is one of the most common multi-agent architectures:

```
Supervisor (coordinator)
├── Decides which agent works next
├── Monitors progress
└── Determines when to finish

Workers (specialists)
├── Researcher → gathers information
├── Writer → creates content
├── Reviewer → evaluates quality
└── Finalizer → polishes output
```

The supervisor uses `Command({ goto })` to route — no conditional edges needed!

### Module 10: Functional API

**Theory: Two Ways to Build**

```
StateGraph API (Declarative):     Functional API (Imperative):
  Define nodes                      Define tasks
  Wire edges                        Call tasks in sequence
  Compile graph                     Use if/else, loops, Promise.all
  Invoke                            Invoke

  Best for: Complex routing,        Best for: Simpler workflows,
  multi-agent, visualization        rapid prototyping, familiarity
```

The Functional API uses `entrypoint()` (your main function) and `task()` (units of work). It feels like writing normal async/await code, but you still get streaming, persistence, and tracing.

### Module 11: Command Routing

**Theory: Command vs Conditional Edges**

```
Old pattern (still valid):
  addConditionalEdges("node", routerFn, ["a", "b", "c"])
  → Router logic is SEPARATE from node logic
  → Must declare ALL possible destinations upfront

New pattern (Command):
  return new Command({ goto: "a", update: { ... } })
  → Routing is INSIDE the node (co-located with logic)
  → Route AND update state in ONE return
  → Use addNode("name", fn, { ends: [...] }) to declare destinations
```

Command is especially powerful for multi-agent systems where agents decide where to hand off control.

### Module 12: Long-term Memory

**Theory:** `InMemoryStore` is a key-value store organized by namespaces:

```
InMemoryStore
├── ["user-123", "preferences"]
│   ├── "theme" → { theme: "dark" }
│   └── "language" → { language: "TypeScript" }
├── ["user-123", "history"]
│   └── "last-topic" → { topic: "LangGraph" }
└── ["user-456", "preferences"]
    └── "theme" → { theme: "light" }
```

Access it inside nodes via `config.store`. This lets your agent learn and remember across completely separate conversations.

### Module 13: Subgraphs (NEW)

**Theory:** Subgraphs let you compose graphs like functions:

```
Two styles:

1. DIFFERENT STATE (wrapper node):
   Parent State: { topic, summary }
   Child State:  { query, analysis }
   → You manually map fields between them
   → Maximum encapsulation, reusable

2. SHARED STATE (compiled graph as node):
   Parent & Child share the SAME Annotation
   → No mapping needed, child reads/writes directly
   → Simpler, but tighter coupling
```

### Module 14: Error Handling (NEW)

**Theory:** AI applications have unique failure modes — LLM API errors, invalid outputs, tool failures, infinite loops. This module covers retry-with-validation, fallback nodes, and a reusable `withRetry()` wrapper with exponential backoff.

### Module 15: Message Trimming (NEW)

**Theory:** As conversations grow, messages accumulate and eventually overflow the LLM's context window. This module covers three strategies: simple count-based trimming, `trimMessages()` with token counting, and the summarize-then-trim approach.

### Module 16: Visualization (NEW)

**Theory:** `graph.getGraph().drawMermaid()` generates a Mermaid diagram of your graph structure. Paste it into GitHub, VS Code, or mermaid.live to visualize your architecture.

---

## Two APIs: StateGraph vs Functional

LangGraph offers two ways to build workflows:

```
┌──────────────────────────┬──────────────────────────┐
│     StateGraph API       │     Functional API       │
├──────────────────────────┼──────────────────────────┤
│ new StateGraph(State)    │ entrypoint("name", fn)   │
│ .addNode("name", fn)     │ task("name", fn)         │
│ .addEdge(A, B)           │ await taskFn(input)      │
│ .addConditionalEdges()   │ if/else, while, for      │
│ .compile()               │ Promise.all() for parallel│
├──────────────────────────┼──────────────────────────┤
│ Declarative              │ Imperative               │
│ Visual (Mermaid)         │ Feels like normal code   │
│ Complex routing          │ Quick prototyping        │
│ Multi-agent systems      │ Simple workflows         │
└──────────────────────────┴──────────────────────────┘
```

**Rule of thumb:** Start with StateGraph. Switch to Functional if your workflow is straightforward and you don't need visualization or complex routing.

---

## Production Considerations

### Checkpointers (Memory Storage)

| Checkpointer | Use Case | Persistence |
|---|---|---|
| `MemorySaver` | Development/testing | In-memory (lost on restart) |
| `PostgresSaver` | Production | PostgreSQL database |
| `SqliteSaver` | Lightweight production | SQLite file |

```typescript
// Development
import { MemorySaver } from "@langchain/langgraph";
const checkpointer = new MemorySaver();

// Production
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
const checkpointer = PostgresSaver.fromConnString("postgresql://...");
```

### Key Production Patterns

1. **Always set max iterations** — Prevent infinite agent loops
2. **Use streaming** — Don't make users wait for the full result
3. **Trim messages** — Manage context window in long conversations
4. **Human-in-the-loop** — Gate dangerous actions behind approval
5. **Error handling** — Retry transient failures, fallback gracefully
6. **Persistent checkpointers** — MemorySaver loses data on restart
7. **Structured output** — Use Zod schemas for reliable LLM output

---

## Learning Path

### Beginner (2-3 hours)
1. `01-graph-basics/` — Understand nodes, edges, state
2. `02-state-management/` — Master Annotations and reducers
3. `03-conditional-edges/` — Dynamic routing and loops

### Intermediate (3-4 hours)
4. `04-tool-integration/` — Give agents tools (ReAct pattern)
5. `05-memory-checkpoints/` — Persistence and state inspection
6. `06-human-in-the-loop/` — interrupt/resume workflows
7. `07-streaming/` — Real-time output

### Advanced (4-5 hours)
8. `08-prebuilt-agents/` — createReactAgent with custom state
9. `10-functional-api/` — Alternative API (entrypoint + task)
10. `11-command-routing/` — Command routing + agent handoffs
11. `12-long-term-memory/` — InMemoryStore cross-thread memory

### Production-Ready (2-3 hours)
12. `13-subgraphs/` — Graph composition
13. `14-error-handling/` — Retry, fallback, resilience
14. `15-message-trimming/` — Context window management
15. `16-visualization/` — Debug with Mermaid diagrams
16. `09-mini-project/` — Full multi-agent pipeline (capstone)

---

## Quick Reference

### Common Imports

```typescript
// Core
import { StateGraph, Annotation, MessagesAnnotation, START, END } from "@langchain/langgraph";

// Persistence
import { MemorySaver, InMemoryStore } from "@langchain/langgraph";

// Human-in-the-loop
import { interrupt, Command } from "@langchain/langgraph";

// Functional API
import { entrypoint, task, getPreviousState, addMessages } from "@langchain/langgraph";

// Prebuilt
import { createReactAgent, ToolNode } from "@langchain/langgraph/prebuilt";

// Tools
import { tool } from "@langchain/core/tools";

// Messages
import { HumanMessage, AIMessage, SystemMessage, trimMessages } from "@langchain/core/messages";
```

### Cheat Sheet

```typescript
// 1. Define state
const State = Annotation.Root({
  field: Annotation<string>(),
  list: Annotation<string[]>({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  }),
});

// 2. Build graph
const graph = new StateGraph(State)
  .addNode("name", async (state) => ({ field: "value" }))
  .addEdge(START, "name")
  .addEdge("name", END)
  .compile({ checkpointer: new MemorySaver() });

// 3. Run
const result = await graph.invoke(
  { field: "input" },
  { configurable: { thread_id: "1" } }
);

// 4. Stream
for await (const update of await graph.stream(input, { streamMode: "updates" })) {
  console.log(update);
}
```

---

**Happy Learning!** Run the examples in order, read the theory in each file, and experiment by modifying the code.
