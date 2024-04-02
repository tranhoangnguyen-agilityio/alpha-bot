from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain.tools.render import render_text_description

from langchain_extensions.agents.fastchat_tools.output_parser import (
    FastChatToolAgentOutputParser,
)
from langchain_extensions.format_scratchpad.fastchat_tools import (
    format_to_fastchat_tool_messages,
)


def create_fastchat_tools_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:

    llm_with_tools = llm.bind(tools=tools)

    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_fastchat_tool_messages(
                x["intermediate_steps"],
            )
        )
        | prompt
        | llm_with_tools
        | FastChatToolAgentOutputParser()
    )

    return agent
