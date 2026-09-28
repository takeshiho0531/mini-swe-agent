class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format.

    model_response retains the rejected assistant response and its usage/cost
    metadata for accounting. It is not added to the conversation as an action.
    """

    def __init__(self, *messages: dict, model_response: dict | None = None):
        super().__init__(*messages)
        self.model_response = model_response
