import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import { GraphRecursionError } from "@langchain/langgraph";
import { describe, expect, it } from "vitest";
import { buildAgent, type ToolCallingModel } from "./agent.ts";

/**
 * A deterministic stand-in for the model: returns the scripted replies in order, repeating the last one.
 * Each call returns a fresh copy. The messages reducer merges messages by id, so reusing one instance
 * would replace the earlier message instead of appending a new one.
 */
function scriptedModel(replies: AIMessage[]): ToolCallingModel & { calls: BaseMessage[][] } {
  const calls: BaseMessage[][] = [];
  return {
    calls,
    async invoke(messages) {
      calls.push(messages);
      const reply = replies[Math.min(calls.length - 1, replies.length - 1)]!;
      return new AIMessage({ content: reply.content, tool_calls: reply.tool_calls });
    },
  };
}

function callTool(id: string, name: string, a: number, b: number) {
  return new AIMessage({ content: "", tool_calls: [{ id, name, args: { a, b }, type: "tool_call" }] });
}

describe("calculator agent", () => {
  it("loops between the model and the tool node until the model answers", async () => {
    const model = scriptedModel([
      callTool("1", "add", 3, 4),
      callTool("2", "multiply", 7, 2),
      new AIMessage("The answer is 14."),
    ]);

    const result = await buildAgent(model).invoke({
      messages: [new HumanMessage("Add 3 and 4, then multiply the result by 2.")],
    });

    expect(result.llmCalls).toBe(3);
    expect(result.messages.map((message) => message.type)).toEqual(["human", "ai", "tool", "ai", "tool", "ai"]);
    expect(result.messages.filter(ToolMessage.isInstance).map((message) => message.text)).toEqual(["7", "14"]);
    expect(result.messages.at(-1)!.text).toBe("The answer is 14.");
    // The second model call sees the first tool result.
    expect(model.calls[1]!.filter(ToolMessage.isInstance).map((message) => message.text)).toEqual(["7"]);
  });

  it("ends after one call when the model answers without a tool", async () => {
    const model = scriptedModel([new AIMessage("Hello.")]);

    const result = await buildAgent(model).invoke({ messages: [new HumanMessage("Hi")] });

    expect(result.llmCalls).toBe(1);
    expect(result.messages.map((message) => message.type)).toEqual(["human", "ai"]);
  });

  it("is stopped by the recursion limit when the model never stops calling tools", async () => {
    const model = scriptedModel([callTool("loop", "add", 1, 1)]);

    await expect(
      buildAgent(model).invoke({ messages: [new HumanMessage("Keep adding.")] }, { recursionLimit: 6 }),
    ).rejects.toBeInstanceOf(GraphRecursionError);
  });
});
