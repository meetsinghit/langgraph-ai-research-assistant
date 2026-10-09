"""Deterministic fakes for tests — no network, no API keys."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field


class ScriptedChatModel(BaseChatModel):
    """Plays back scripted AIMessages and structured outputs.

    - ``responses``:          AIMessages returned by plain ``invoke``
    - ``structured_outputs``: objects returned by
                              ``with_structured_output(...).invoke``
    - ``raise_on_call``:      call index -> exception to raise
                              (provider failures, rate limits)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: list[AIMessage] = Field(default_factory=list)
    structured_outputs: list[Any] = Field(default_factory=list)
    raise_on_call: dict[int, Exception] = Field(default_factory=dict)
    call_index: int = 0
    structured_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = self.call_index
        self.call_index += 1
        if index in self.raise_on_call:
            raise self.raise_on_call[index]
        if self.responses:
            message = self.responses[min(index, len(self.responses) - 1)]
        else:
            message = AIMessage(content="")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        outputs = self.structured_outputs

        def _invoke(_messages: Any) -> Any:
            if not outputs:
                raise RuntimeError("no scripted structured output available")
            index = min(self.structured_index, len(outputs) - 1)
            self.structured_index += 1
            return outputs[index]

        return RunnableLambda(_invoke)


class FakeTools:
    """Deterministic stand-in for ResearchTools."""

    def __init__(
        self,
        search_results: list[dict[str, str]] | None = None,
        page_text: str = "",
    ):
        self.search_results = search_results or []
        self.page_text = page_text
        self.search_calls: list[str] = []
        self.fetch_calls: list[str] = []

    def web_search(self, query: str) -> list[dict[str, str]]:
        self.search_calls.append(query)
        return list(self.search_results)

    def fetch_page(self, url: str) -> str:
        self.fetch_calls.append(url)
        return self.page_text

    def tool_functions(self) -> dict[str, Any]:
        return {"web_search": self.web_search, "fetch_page": self.fetch_page}

    def langchain_tools(self) -> list[Any]:
        return []  # the scripted model ignores tool schemas
