# app/utils/telemetry.py
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class TurnLog:
    route: str
    router_conf: float
    prompt_chars: int
    ctx_chars: int
    latency_ms: int
    session_id: str = ""
    user_input_chars: int = 0
    error: str = ""

class TurnTimer:
    def __enter__(self):
        self.t0 = time.time()
        return self
    
    def __exit__(self, *args):
        self.ms = int((time.time() - self.t0) * 1000)

def log_turn_metrics(turn_log: TurnLog):
    """Log turn metrics for monitoring and evaluation"""
    logger.info(
        f"Turn metrics - Route: {turn_log.route}, "
        f"Confidence: {turn_log.router_conf:.3f}, "
        f"Latency: {turn_log.latency_ms}ms, "
        f"Input chars: {turn_log.user_input_chars}, "
        f"Prompt chars: {turn_log.prompt_chars}, "
        f"Context chars: {turn_log.ctx_chars}"
        + (f", Error: {turn_log.error}" if turn_log.error else "")
    )
