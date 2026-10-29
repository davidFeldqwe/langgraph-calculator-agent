import { AIMessage, SystemMessage, type BaseMessage, type ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import {
  END,
  MessagesValue,
  ReducedValue,
  START,
  StateGraph,
  StateSchema,
  type ConditionalEdgeRouter,
  type GraphNode,
} from "@langchain/langgraph";
import * as z from "zod";

const numbers = z.object({
  a: z.number().describe("First number"),
  b: z.number().describe("Second number"),
});

const add = tool(({ a, b }) => a + b, { name: "add", description: "Add two numbers", schema: numbers });
const multiply = tool(({ a, b }) => a * b, { name: "multiply", description: "Multiply two numbers", schema: numbers });
const divide = tool(({ a, b }) => a / b, { name: "divide", description: "Divide two numbers", schema: numbers });

export const toolsByName = {
  [add.name]: add,
  [multiply.name]: multiply,
  [divide.name]: divide,
};
export const tools = Object.values(toolsByName);

/** Anything that turns messages into an AI reply: a chat model after `bindTools(tools)`, or a test stub. */
export interface ToolCallingModel {
  invoke(messages: BaseMessage[]): Promise<AIMessage>;
}

export const MessagesState = new StateSchema({
  messages: MessagesValue,
  llmCalls: new ReducedValue(z.number().default(0), { reducer: (x, y) => x + y }),
});

export function buildAgent(modelWithTools: ToolCallingModel) {
  const llmCall: GraphNode<typeof MessagesState> = async (state) => {
    const response = await modelWithTools.invoke([
      new SystemMessage("You are a helpful assistant tasked with performing arithmetic on a set of inputs."),
      ...state.messages,
    ]);
    return { messages: [response], llmCalls: 1 };
  };

  const toolNode: GraphNode<typeof MessagesState> = async (state) => {
    const lastMessage = state.messages.at(-1);
    if (lastMessage == null || !AIMessage.isInstance(lastMessage)) {
      return { messages: [] };
    }

    const result: ToolMessage[] = [];
    for (const toolCall of lastMessage.tool_calls ?? []) {
      const tool = toolsByName[toolCall.name as keyof typeof toolsByName];
      if (!tool) throw new Error(`Unknown tool: ${toolCall.name}`);
      result.push(await tool.invoke(toolCall));
    }
    return { messages: result };
  };

  // Route to the tool node while the model keeps asking for tools; otherwise end.
  const shouldContinue: ConditionalEdgeRouter<{ InputSchema: typeof MessagesState; Nodes: "toolNode" }> = (state) => {
    const lastMessage = state.messages.at(-1);
    if (!lastMessage || !AIMessage.isInstance(lastMessage)) return END;
    return lastMessage.tool_calls?.length ? "toolNode" : END;
  };

  return new StateGraph(MessagesState)
    .addNode("llmCall", llmCall)
    .addNode("toolNode", toolNode)
    .addEdge(START, "llmCall")
    .addConditionalEdges("llmCall", shouldContinue, ["toolNode", END])
    .addEdge("toolNode", "llmCall")
    .compile();
}
