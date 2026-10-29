import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { buildAgent, tools } from "./agent.ts";

const question = process.argv.slice(2).join(" ") || "Add 3 and 4, then multiply the result by 2.";

const model = new ChatAnthropic({ model: "claude-sonnet-4-6", temperature: 0 });
const agent = buildAgent(model.bindTools(tools));

const result = await agent.invoke({ messages: [new HumanMessage(question)] });

for (const message of result.messages) {
  console.log(`[${message.type}]: ${message.text}`);
}
console.log(`\nllmCalls: ${result.llmCalls}`);
