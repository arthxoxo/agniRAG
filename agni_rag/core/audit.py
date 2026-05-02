from __future__ import annotations

import json
import time
from typing import Any


class AuditLogger:
    def __init__(self, log_path: str) -> None:
        self._log_path = log_path

    def log(self, event: dict[str, Any]) -> None:
        event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
        with open(self._log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
