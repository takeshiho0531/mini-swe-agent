import litellm

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_text import (
    count_regex_actions,
    format_observation_messages,
    parse_regex_actions,
)


class LitellmTextbasedModelConfig(LitellmModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""
    unclosed_think_fallback: bool = True
    """Accept a reply whose thinking span was never closed.

    Servers running a reasoning parser (e.g. vLLM ``--reasoning-parser qwen3``)
    file everything before ``</think>`` under ``reasoning_content``. A thinking
    model that writes its answer without ever emitting ``</think>`` therefore
    arrives with ``finish_reason == "stop"``, empty ``content`` and the whole
    answer in ``reasoning_content``. Without a reasoning parser that same text
    would have been the content and its action would have been executed. When
    this flag is on and such a reply holds exactly one action block, the action
    is taken from ``reasoning_content`` and that text becomes the message
    content (``extra.action_source == "reasoning_content"`` marks it). Replies
    cut off by the token limit (``finish_reason == "length"``) and replies with
    zero or several action blocks stay format errors.
    """


class LitellmTextbasedModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmTextbasedModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _action_text(self, content: str | None, reasoning_content: str | None, finish_reason: str | None) -> tuple[str, str]:
        """Return (text to parse actions from, its source field).

        The source is ``"content"`` unless the reply matches the unclosed-think
        signature described on ``unclosed_think_fallback``.
        """
        content = content or ""
        if (
            self.config.unclosed_think_fallback
            and finish_reason == "stop"
            and not content.strip()
            and reasoning_content
            and count_regex_actions(reasoning_content, action_regex=self.config.action_regex) == 1
        ):
            return reasoning_content, "reasoning_content"
        return content, "content"

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse actions from the model response. Raises FormatError if not exactly one action."""
        choice = response.choices[0]
        text, _ = self._action_text(
            choice.message.content, getattr(choice.message, "reasoning_content", None), choice.finish_reason
        )
        return parse_regex_actions(
            text, action_regex=self.config.action_regex, format_error_template=self.config.format_error_template
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        message = super().query(messages, **kwargs)
        choices = message["extra"].get("response", {}).get("choices") or [{}]
        text, source = self._action_text(
            message.get("content"), message.get("reasoning_content"), choices[0].get("finish_reason")
        )
        if source == "reasoning_content":
            # Store the reply the way a server without a reasoning parser would
            # have returned it, so the history the model sees is the same as in
            # runs served without the parser. The raw response is kept in
            # extra["response"].
            message["content"] = text
            message.pop("reasoning_content", None)
            message["extra"]["action_source"] = source
        return message

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into observation messages."""
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
