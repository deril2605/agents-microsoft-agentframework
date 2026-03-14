# Agent Framework Study Guide

This repo is my running guide to learning how to build agents with Microsoft Agent Framework.

The focus here is not just "what code was written", but the concepts behind agent design:
- how agents are structured
- when tools are useful
- how memory and context work
- how to keep context small and relevant
- how to reason about retrieval, middleware, and delegation

Right now this guide covers the first 2 days of study:
- Day 1: Building agents
- Day 2: Context and memory

As I keep studying, I can extend this README day by day.

## Repo Map

- Day 1: [scripts/01-building-agents](./scripts/01-building-agents)
- Day 2: [scripts/02-context-and-memory](./scripts/02-context-and-memory)

Slide decks:
- Building agents slides: [scripts/01-building-agents/README.md](./scripts/01-building-agents/README.md)
- Context and memory slides: [scripts/02-context-and-memory/README.md](./scripts/02-context-and-memory/README.md)

## Core Mental Model

An agent is usually made up of a few building blocks:

- `Model`
  The LLM that reasons, plans, decides, and writes responses.

- `Instructions`
  The role, rules, and behavior expectations for the agent.

- `Tools`
  Functions or external capabilities the model can call when it needs to act or fetch data.

- `Session`
  The object that keeps a conversation tied together across turns.

- `Context Providers`
  Components that inject or store context before and after a run.

- `Middleware`
  Logic that wraps the execution pipeline and can modify behavior before or after the agent runs.

- `Sub-agents`
  Other agents used as specialized workers so the top-level agent stays simpler and smaller.

## Day 1: Building Agents

Folder: [scripts/01-building-agents](./scripts/01-building-agents)

### What I learned

#### 1. A basic agent is just model + instructions

Script: [01-agent-basic.py](./scripts/01-building-agents/01-agent-basic.py)

The simplest usable agent does not need orchestration complexity. At minimum:
- create a client
- create an `Agent`
- give it instructions
- run it on user input

This is the baseline for everything else.

#### 2. Tool calling is the bridge from "chat" to "action"

Scripts:
- [02-tool-calling-without-framework.py](./scripts/01-building-agents/02-tool-calling-without-framework.py)
- [03-agent-tool.py](./scripts/01-building-agents/03-agent-tool.py)
- [04-agent-tools.py](./scripts/01-building-agents/04-agent-tools.py)

Big takeaway:
- a model alone can reason
- a model with tools can do things

Important concepts:
- tool schemas matter because they define what the model is allowed to call
- single-tool agents are easy to reason about
- multi-tool agents need clearer instructions so the model chooses the right tool

Good interview-level understanding:
- tool calling is structured delegation to code
- the model decides when to call, but the tool implementation decides what actually happens
- tool quality depends on naming, descriptions, and argument clarity

#### 3. MCP extends agents beyond local functions

Scripts:
- [05-agent-mcp-local.py](./scripts/01-building-agents/05-agent-mcp-local.py)
- [06-agent-mcp-remote.py](./scripts/01-building-agents/06-agent-mcp-remote.py)

MCP matters because it standardizes how agents connect to external capabilities.

What to remember:
- local MCP is useful when the capability is on the same machine or tightly controlled
- remote MCP is useful when the capability lives elsewhere and should be exposed as a service
- MCP is about interoperability, not just another way to write tools

#### 4. Middleware gives you control over the execution pipeline

Script: [07-agent-middleware.py](./scripts/01-building-agents/07-agent-middleware.py)

Middleware is where cross-cutting behavior belongs.

Examples of what middleware is good for:
- logging
- retries
- guardrails
- policy enforcement
- cost tracking
- input/output transformations

Key distinction:
- tools solve task-specific actions
- middleware solves system-wide execution behavior

#### 5. Supervisors are for orchestration

Script: [08-agent-supervisor.py](./scripts/01-building-agents/08-agent-supervisor.py)

A supervisor pattern becomes useful when one agent should coordinate other specialized agents instead of doing everything itself.

This introduces an important design shift:
- not every problem should be handled by one giant agent
- specialization often beats one-agent-does-everything designs

### Day 1 Summary

By the end of Day 1, the main ideas are:
- start simple
- give agents tools when they need to act
- use MCP when capabilities should be standardized or externalized
- use middleware for system behavior
- use supervision when specialization or coordination is needed

## Day 2: Context And Memory

Folder: [scripts/02-context-and-memory](./scripts/02-context-and-memory)

This day is really about one central question:

How does an agent remember the right things without carrying everything forever?

### 1. Sessions are the first level of memory

Script: [01-agent-session.py](./scripts/02-context-and-memory/01-agent-session.py)

Core idea:
- without a session, each run is isolated
- with a session, the agent can maintain conversational continuity

What matters:
- session memory is the simplest form of conversational state
- it is ideal for short-lived interactions
- by itself, session state is usually process-local unless you back it with external storage

### 2. History providers persist conversation history

Scripts:
- [02-agent-history-using-redis.py](./scripts/02-context-and-memory/02-agent-history-using-redis.py)
- [03-agent-history-using-sqllite-custom.py](./scripts/02-context-and-memory/03-agent-history-using-sqllite-custom.py)

Core idea:
- history providers save and replay prior messages

When to use:
- when you want a session to survive restarts
- when multiple runs or processes should resume the same conversation

Difference from basic session memory:
- session memory is runtime continuity
- history providers are persistence for that continuity

Redis version:
- good for shared, external, service-style storage

SQLite version:
- good for simple, file-based local persistence

### 3. Dynamic memory is not just chat history

Scripts:
- [04-agent-dynamic-memory-redis.py](./scripts/02-context-and-memory/04-agent-dynamic-memory-redis.py)
- [05-agent-dynamic-memory-mem0.py](./scripts/02-context-and-memory/05-agent-dynamic-memory-mem0.py)

Dynamic memory answers a different problem:
- not "what was every prior message?"
- but "what prior information is worth bringing back now?"

Two important flavors:

#### Redis dynamic memory

Script: [04-agent-dynamic-memory-redis.py](./scripts/02-context-and-memory/04-agent-dynamic-memory-redis.py)

This stores searchable conversational content in Redis and retrieves relevant memories later.

Best mental model:
- store lots of context externally
- fetch only the relevant parts per query

#### Mem0 dynamic memory

Script: [05-agent-dynamic-memory-mem0.py](./scripts/02-context-and-memory/05-agent-dynamic-memory-mem0.py)

This is more distilled memory:
- it tries to extract durable facts from conversation
- then retrieve those facts later

Best mental model:
- history is full transcript
- dynamic memory is selectively remembered information

### 4. Knowledge retrieval is different from memory

Scripts:
- [06-agent-knowledge-sqllite-full-text-search.py](./scripts/02-context-and-memory/06-agent-knowledge-sqllite-full-text-search.py)
- [07-agent-knowledge-postgres-hybrid-search.py](./scripts/02-context-and-memory/07-agent-knowledge-postgres-hybrid-search.py)
- [08-agent-knowledge-hybrid-query-rewrite-pg.py](./scripts/02-context-and-memory/08-agent-knowledge-hybrid-query-rewrite-pg.py)

This is a very important distinction.

Memory is about the conversation or user.
Knowledge is about external domain data.

#### SQLite full-text retrieval

Script: [06-agent-knowledge-sqllite-full-text-search.py](./scripts/02-context-and-memory/06-agent-knowledge-sqllite-full-text-search.py)

This is keyword-based retrieval.

Good for:
- small corpora
- structured local data
- simple demos
- cases where semantic search is not needed

#### PostgreSQL hybrid search

Script: [07-agent-knowledge-postgres-hybrid-search.py](./scripts/02-context-and-memory/07-agent-knowledge-postgres-hybrid-search.py)

This combines:
- vector similarity search
- full-text search

Why this matters:
- vector search is good for meaning
- keyword search is good for exact terms
- hybrid search combines both strengths

Important concept:
- retrieval quality often improves when semantic and lexical signals are fused

#### Query rewrite before retrieval

Script: [08-agent-knowledge-hybrid-query-rewrite-pg.py](./scripts/02-context-and-memory/08-agent-knowledge-hybrid-query-rewrite-pg.py)

This solves the follow-up question problem.

Example:
- user asks a vague follow-up like "what about lighter options?"
- latest message alone is not enough
- so the system rewrites the whole conversation into a better retrieval query

Key concept:
- retrieval can fail not because the knowledge base is bad, but because the search query is weak
- rewriting is a retrieval optimization technique

### 5. Middleware can be used for context compaction

Script: [09-agent-summarization-middleware.py](./scripts/02-context-and-memory/09-agent-summarization-middleware.py)

This script shows summarization as a middleware pattern.

Core idea:
- long histories become too large or expensive
- summarize earlier turns
- replace detailed history with a compact summary

When this matters:
- long-running assistants
- constrained context windows
- cost-sensitive systems

Key mental model:
- not all prior context deserves verbatim preservation
- summaries are a form of compressed memory

### 6. Sub-agents help isolate heavy context

Scripts:
- [10-agent-with-subagents.py](./scripts/02-context-and-memory/10-agent-with-subagents.py)
- [11-agent-without-subagents.py](./scripts/02-context-and-memory/11-agent-without-subagents.py)

These two are best understood as a pair.

#### Without sub-agents

Script: [11-agent-without-subagents.py](./scripts/02-context-and-memory/11-agent-without-subagents.py)

One agent reads files, searches files, and keeps all raw outputs in one context.

Problem:
- context gets bloated
- token cost rises
- the top-level reasoning agent gets noisy inputs

#### With sub-agents

Script: [10-agent-with-subagents.py](./scripts/02-context-and-memory/10-agent-with-subagents.py)

The top-level coordinator delegates file-heavy work to a research sub-agent.

Benefit:
- sub-agent absorbs raw tool output
- coordinator only sees concise summarized findings
- top-level context stays smaller and cleaner

Key concept:
- sub-agents are not just about specialization
- they are also a context management strategy

### Day 2 Summary

By the end of Day 2, the main ideas are:
- sessions give continuity
- history providers give persistence
- dynamic memory retrieves relevant prior information
- knowledge retrieval grounds answers in external data
- query rewriting improves retrieval in multi-turn settings
- summarization compresses context
- sub-agents quarantine heavy context

## Concepts To Be Able To Explain Clearly

These are the concepts I should be able to explain simply and confidently.

### Agent basics

- What is the difference between an LLM call and an agent?
- Why are tools needed?
- What makes a tool schema good or bad?
- When do you use a supervisor or sub-agent instead of one agent?

### Context and memory

- Difference between session, history, memory, and knowledge
- Difference between chat transcript persistence and semantic memory
- Why not keep all past messages forever?
- Why retrieval is often better than naive long-context stuffing

### Retrieval

- Keyword search vs vector search
- Why hybrid retrieval is useful
- Why query rewriting helps in follow-up questions
- Why retrieval should often happen before the model answers

### Middleware and control

- Why middleware is a better home for cross-cutting behavior
- How summarization middleware helps long conversations
- How middleware differs from tools and context providers

### Multi-agent design

- When sub-agents are helpful
- How sub-agents reduce context bloat
- Why total tokens may still be high even when coordinator tokens are lower

## Quick Comparison Table

| Concept | Main Purpose | Best Mental Model | Example |
| --- | --- | --- | --- |
| Session | Keep turns connected | short-term conversational continuity | [01-agent-session.py](./scripts/02-context-and-memory/01-agent-session.py) |
| History Provider | Persist transcript | stored conversation replay | [02-agent-history-using-redis.py](./scripts/02-context-and-memory/02-agent-history-using-redis.py) |
| Dynamic Memory | Retrieve relevant prior info | remembered facts or searchable memories | [05-agent-dynamic-memory-mem0.py](./scripts/02-context-and-memory/05-agent-dynamic-memory-mem0.py) |
| Knowledge Provider | Inject external domain facts | RAG for grounding | [07-agent-knowledge-postgres-hybrid-search.py](./scripts/02-context-and-memory/07-agent-knowledge-postgres-hybrid-search.py) |
| Middleware | Wrap execution behavior | cross-cutting pipeline control | [09-agent-summarization-middleware.py](./scripts/02-context-and-memory/09-agent-summarization-middleware.py) |
| Sub-agent | Isolate work and context | delegated worker with its own context window | [10-agent-with-subagents.py](./scripts/02-context-and-memory/10-agent-with-subagents.py) |

## Practical Design Heuristics

- Start with one agent before designing many.
- Add tools only when the model must act or fetch data.
- Use sessions for continuity, but add persistence only when needed.
- Treat memory and knowledge as different systems.
- Prefer retrieval over dumping huge raw context into the model.
- Use middleware for concerns that should apply to every run.
- Use sub-agents when raw data is large, noisy, or specialized.
- If a follow-up question is ambiguous, improve the search query before blaming retrieval.

## What To Add Next

As more study days are added, this README can grow with the same structure:
- day summary
- concepts learned
- notable scripts
- design takeaways
- quick comparison notes

For now, the best places to review are:
- [scripts/01-building-agents](./scripts/01-building-agents)
- [scripts/02-context-and-memory](./scripts/02-context-and-memory)
