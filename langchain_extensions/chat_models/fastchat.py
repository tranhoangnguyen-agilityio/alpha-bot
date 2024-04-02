from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    cast,
)
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor, RunnableConfig
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
)


class AuthBaseMessage(BaseMessage):
    auth_config: str = ""


class AuthBaseMessageChunk(BaseMessageChunk):
    auth_config: str = ""


class FastChat(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = FastChat(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    n: int
    """The number of characters from the last message of the prompt to be echoed."""

    model_name: str = Field(default="lmsys/fastchat-t5-3b-v1.0", alias="model")
    """Model name to use"""

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        # Initialize a model. Not for sure this is the best place to load the model local.
        if not values.get("client"):
            model_name = "lmsys/fastchat-t5-3b-v1.0"
            values["client"] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if not values.get("tokenizer"):
            model_name = "lmsys/fastchat-t5-3b-v1.0"
            values["tokenizer"] = T5Tokenizer.from_pretrained(
                model_name, use_fast=False
            )

        return values

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        message = super().invoke(input=input, config=config, stop=stop, **kwargs)
        auth_message = cast(AuthBaseMessage, message)
        auth_message.auth_config = self._get_auth_config_from_message(input)
        return auth_message

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        for message_chunk in super().stream(input, config, stop=stop, **kwargs):
            auth_message_chunk = cast(AuthBaseMessageChunk, message_chunk)
            auth_message_chunk.auth_config = self._get_auth_config_from_message(input)
            yield auth_message_chunk

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        """Override to extract and store the auth_config"""
        async for message_chunk in super().astream(input, config, stop=stop, **kwargs):
            auth_message_chunk = cast(AuthBaseMessageChunk, message_chunk)
            auth_message_chunk.auth_config = self._get_auth_config_from_message(input)
            yield auth_message_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]

        # Invoke model to generate the completion
        inputs = self.tokenizer(last_message.content, return_tensors="pt")
        output = self.client.generate(inputs["input_ids"], max_new_tokens=100)[0]
        response = self.tokenizer.decode(output, skip_special_tokens=True)

        print("RESPONSE: ", response)
        # Pass the response to the message output
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]

        print("The _stream function", last_message.content)

        # Invoke model to generate the completion
        inputs = self.tokenizer(last_message.content, return_tensors="pt")

        for token in self.client.generate(inputs["input_ids"], max_new_tokens=100)[0]:
            decoded_token = self.tokenizer.decode(token, skip_special_tokens=True)
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=decoded_token))

            if run_manager:
                run_manager.on_llm_new_token(decoded_token, chunk=chunk)

            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """An async variant of astream.

        If not provided, the default behavior is to delegate to the _generate method.

        The implementation below instead will delegate to `_stream` and will
        kick it off in a separate thread.

        If you're able to natively support async, then by all means do so!
        """

        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            yield chunk

    def _assign_fastchat_model(self, fast_chat_model):
        """Assign the fastchat model locally"""
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def _get_auth_config_from_message(self, input: LanguageModelInput):
        """Get the auth_config property from the input if any. Otherwise return None"""
        if type(input) is not str and getattr(input, "dict") is not None:
            return input.dict().get("auth_config")
        return ""

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"n": self.n}
