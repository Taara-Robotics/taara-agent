import struct

from .agent_message import AgentMessage


class FrameMessage(AgentMessage):
    VERSION = 1
    TYPE = 2

    def __init__(self, payload: bytes):
        super().__init__(payload)
