class AgentMessage:
    VERSION: int
    TYPE: int

    def __init__(self, payload: bytes):
        self._payload = payload

    @property
    def payload(self):
        return self._payload
