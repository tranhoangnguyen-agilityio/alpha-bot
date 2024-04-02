# Refer from libs/langchain/langchain/agents/output_parsers/openai_tools.py
import re
import random
from typing import Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)

from langchain.agents.agent import MultiActionAgentOutputParser

from langchain.output_parsers import RegexParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import run_in_executor, RunnableConfig

from langchain_core.runnables.utils import (
    Input,
    Output,
)

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator


class FastChatToolAgentAction(AgentActionMessageLog):
    tool_call_id: str
    """Tool call that this message is responding to."""


class ActionInputRegexParser(BaseOutputParser):
    """Parse the output of an LLM call using a regex."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    regex: str
    """The regex to use to parse the output."""
    output_keys: List[str]
    """The keys to use for the output."""
    default_output_key: Optional[str] = None
    """The default key to use for the output."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_parser"

    def parse(self, text: str) -> Dict[str, str]:
        """Parse the  output of an LLM call."""
        matches = re.findall(self.regex, text)
        input = {}
        for item in matches:
            k, v = item.split(":")
            input[k.strip()] = v.strip()
        return input


def parse_ai_message_to_fastchat_tool_action(
    message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls"""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    actions: List = []

    content_msg = f"responded: {message.content}\n" if message.content else "\n"

    # Parse LLM completion to retrieve the action and action input
    response = re.sub(" +", " ", message.content)

    action_parser = RegexParser(
        regex=r"Action: (.*)",
        output_keys=["action"],
    )
    action_input_parser = RegexParser(
        regex=r"Action Input: (.*)",
        output_keys=["action_input"],
    )

    dict1 = {}
    dict2 = {}
    try:
        dict1 = action_parser.parse(response)
        dict2 = action_input_parser.parse(response)
    except ValueError as e:
        pass
    except Exception as e:
        raise e

    function_name = dict1.get("action") or None

    tool_input = {}

    if action_input_str := dict2.get("action_input"):
        parser = ActionInputRegexParser(regex=r"\b(\w+\s*:\s*\w+)\b", output_keys=[])
        tool_input = parser.parse(action_input_str)
        # tool_input = {
        #    "order_id": 1, "auth_config": "test"}

    # Append the auth_config if exist
    tool_input["auth_config"] = message.additional_kwargs.get("auth_config")

    if function_name:
        random.seed(10)

        tool_call_id = str(random.random())
        log = f"\nInvoking: `{function_name}` with `{tool_input}`"

        actions.append(
            FastChatToolAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
                tool_call_id=tool_call_id,
            )
        )
    else:
        #  Finish
        return AgentFinish(
            return_values={"output": message.content}, log=str(message.content)
        )
    return actions


class FastChatToolAgentOutputParser(MultiActionAgentOutputParser):

    @property
    def _type(self) -> str:
        return "fastchat-tools-agent-output-parser"

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        print(FastChatToolAgentOutputParser)
        print("---------------------BEGIN------------")
        print(message.content)
        print("---------------------END------------")
        return parse_ai_message_to_fastchat_tool_action(message)

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """
        Default implementation of transform, which buffers input and then calls stream.
        Subclasses should override this method if they can start producing output while
        input is still being generated.
        """
        final: Input
        got_first_val = False

        for chunk in input:
            if not got_first_val:
                final = chunk
                final.additional_kwargs["auth_config"] = chunk.dict().get("auth_config")
                got_first_val = True
            else:
                # Make a best effort to gather, for any type that supports `+`
                # This method should throw an error if gathering fails.
                final = final + chunk  # type: ignore[operator]

        if got_first_val:
            yield from self.stream(final, config, **kwargs)


class ActionInputRegexParser(BaseOutputParser):
    """Parse the output of an LLM call using a regex."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    regex: str
    """The regex to use to parse the output."""
    output_keys: List[str]
    """The keys to use for the output."""
    default_output_key: Optional[str] = None
    """The default key to use for the output."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_parser"

    def parse(self, text: str) -> Dict[str, str]:
        """Parse the  output of an LLM call."""
        matches = re.findall(self.regex, text)
        input = {}
        for item in matches:
            k, v = item.split(":")
            input[k.strip()] = v.strip()
        return input
