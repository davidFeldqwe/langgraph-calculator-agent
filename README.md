# langgraph-calculator-agent

A minimal [LangGraph.js](https://docs.langchain.com/oss/javascript/langgraph/overview) agent in TypeScript, with no build step. It follows the official [LangGraph.js quickstart](https://docs.langchain.com/oss/javascript/langgraph/quickstart), Graph API path. The agent does arithmetic with three tools: `add`, `multiply` and `divide`.

The agent is a `StateGraph` with two nodes:

```
START → llmCall ─(tool calls?)→ toolNode → llmCall … ─(no tool calls)→ END
```

- `llmCall` sends the messages to the model, which is bound to the tools.
- `toolNode` runs each tool call from the last AI message and appends a `ToolMessage` for each result.
- A conditional edge sends the graph back to `toolNode` while the model keeps calling tools. When it answers without a tool call, the graph ends.

The state has two fields. `messages` uses the `MessagesValue` reducer, which appends. `llmCalls` is a counter using a `ReducedValue` that sums.

## Run it

```sh
pnpm install
cp .env.example .env   # set ANTHROPIC_API_KEY in that file, or export it
pnpm start "Add 3 and 4, then multiply the result by 2."
```

It prints every message in the final state and the number of model calls. The model is `claude-sonnet-4-6`, as in the quickstart.

```sh
pnpm test
```

Tests swap the model for a scripted stand-in, so they run without an API key. They check three things:

- The graph loops model → tool → model until the model answers.
- A reply with no tool call ends the graph after one model call.
- A model that never stops calling tools is stopped by LangGraph's `recursionLimit` with a `GraphRecursionError`.

## Differences from the quickstart

- The graph is built by `buildAgent(modelWithTools)`, not at module top level. That way tests can pass in a stub model.
- `toolNode` throws on an unknown tool name. The quickstart would crash on `undefined.invoke`.
- The stub returns a fresh `AIMessage` on each call. `MessagesValue` merges messages by id, so returning the same instance twice would overwrite the earlier message instead of appending one.

Versions: `@langchain/langgraph` 1.5, `@langchain/core` 1.3, `@langchain/anthropic` 1.6.
