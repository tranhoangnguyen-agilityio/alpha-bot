from typing import Any
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools.render import render_text_description
from langchain_core.prompt_values import PromptValue, ChatPromptValue


REACT_INSTRUCTIONS = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}"""


class AuthChatPromptValue(ChatPromptValue):
    """Chat prompt value.

    A type of a prompt value that is built from messages with additional auth_config field.
    """

    auth_config: str = ""


# Custom prompt template class to assign the authentication config
class ChatPromptTemplateWithAuth(ChatPromptTemplate):
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """
        Format prompt. Should return a PromptValue.
        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            PromptValue.
        """
        messages = self.format_messages(**kwargs)
        prompt_value = AuthChatPromptValue(messages=messages)
        prompt_value.auth_config = kwargs.get("auth_config", None)
        return prompt_value


def create_fastchat_prompt():
    return ChatPromptTemplateWithAuth.from_template(REACT_INSTRUCTIONS)
