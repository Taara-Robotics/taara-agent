import struct

from .agent_message import AgentMessage


class IntroductionMessage(AgentMessage):
    VERSION = 1
    TYPE = 1

    def __init__(self, payload: bytes):
        super().__init__(payload)
