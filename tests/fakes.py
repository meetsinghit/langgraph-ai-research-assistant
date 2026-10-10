"""Deterministic fakes for tests — no network, no API keys."""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field

from agent.tools import ToolError


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
    #: messages observed on each plain ``invoke`` — lets tests assert
    #: prompt content (grounding, provenance) and that internal
    #: model messages retain their structured content blocks.
    seen_messages: list[list[Any]] = Field(default_factory=list)
    #: messages observed on each ``with_structured_output`` invoke
    #: (plan and review prompts) — kept separate from
    #: ``seen_messages`` so the positional indices of plain
    #: invocations used by existing tests stay stable.
    seen_structured_messages: list[list[Any]] = Field(default_factory=list)
    #: tool names passed to each ``bind_tools`` call — lets tests
    #: assert the deterministic search/fetch tool partitioning
    #: (which tools the graph made available in each round).
    tool_bind_history: list[list[str]] = Field(default_factory=list)

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
        self.seen_messages.append(list(messages))
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
        self.tool_bind_history.append(
            [str(getattr(tool, "name", tool)) for tool in tools]
        )
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        outputs = self.structured_outputs

        def _invoke(messages: Any) -> Any:
            # recorded separately from seen_messages so tests can
            # assert structured-node prompts (plan, review) without
            # shifting plain-invoke positional indices
            if isinstance(messages, list):
                self.seen_structured_messages.append(list(messages))
            if not outputs:
                raise RuntimeError("no scripted structured output available")
            index = min(self.structured_index, len(outputs) - 1)
            self.structured_index += 1
            return outputs[index]

        return RunnableLambda(_invoke)


class _StubTool:
    """Minimal tool stub exposing the ``name`` the graph filters on."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name


class FakeTools:
    """Deterministic stand-in for ResearchTools."""

    def __init__(
        self,
        search_results: list[dict[str, str]] | None = None,
        page_text: str = "Fetched page content.",
        fail_queries: set[str] | None = None,
        fail_fetch: bool = False,
        fail_fetch_urls: set[str] | None = None,
    ):
        self.search_results = search_results or []
        self.page_text = page_text
        #: queries that should raise (e.g. empty result set) so a
        #: test can exercise the agent's recovery from a bad
        #: query.
        self.fail_queries = fail_queries or set()
        #: when True every fetch_page call raises, so a test can
        #: exercise the unverified-answer path.
        self.fail_fetch = fail_fetch
        #: specific URLs whose fetch fails (e.g. the official page
        #: failing TLS verification) while other fetches succeed —
        #: reproduces the live-run mixed-outcome pass.
        self.fail_fetch_urls = fail_fetch_urls or set()
        self.search_calls: list[str] = []
        self.fetch_calls: list[str] = []

    def web_search(self, query: str) -> list[dict[str, str]]:
        self.search_calls.append(query)
        if query in self.fail_queries:
            raise ToolError(f"web_search returned no results for query {query!r}")
        return list(self.search_results)

    def fetch_page(self, url: str) -> str:
        self.fetch_calls.append(url)
        if self.fail_fetch or url in self.fail_fetch_urls:
            raise ToolError(
                f"fetch_page request failed for {url}: ConnectError: "
                "self-signed certificate in certificate chain. TLS "
                "certificate verification failed; verification is "
                "never disabled."
            )
        return self.page_text

    def tool_functions(self) -> dict[str, Any]:
        return {"web_search": self.web_search, "fetch_page": self.fetch_page}

    def langchain_tools(self) -> list[Any]:
        # stubs: the scripted model ignores schemas, but the graph
        # filters these by name to partition search/fetch phases.
        return [_StubTool("web_search"), _StubTool("fetch_page")]
