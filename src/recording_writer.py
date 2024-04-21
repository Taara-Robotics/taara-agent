import struct
import time

from .agent_message import AgentMessage


class RecordingWriter:
    def __init__(self, filename: str, is_simulation=False, description: str=""):
        self._filename = filename
        self._file = open(self._filename, "wb")

        # write header (recording version 2 as uint8)
        self._file.write("TRR".encode("ascii"))
        self._file.write(struct.pack("<B", 2))

        # Write current timestamp
        self._file.write(struct.pack("<I", round(time.time())))

        # Write is_simulation bool
        self._file.write(struct.pack("<B", 1 if is_simulation else 0))

        # Write description
        description_utf8 = description.encode("UTF-8")
        self._file.write(struct.pack("<I", len(description_utf8)))
        self._file.write(description_utf8)


    def write(self, connection_id: int, message: AgentMessage):
        # write connection id (as uint32)
        self._file.write(struct.pack("<I", connection_id))

        # write message version (as uint16)
        self._file.write(struct.pack("<H", message.VERSION))

        # write message type (as uint16)
        self._file.write(struct.pack("<H", message.TYPE))

        # write message payload size (as uint32)
        self._file.write(struct.pack("<I", len(message.payload)))

        # write message payload bytes
        self._file.write(message.payload)

    def close(self):
        self._file.close()
