"""Framework-native adapters that convert third-party agent run outputs
into tracecheck `Trace` objects.

Each adapter is duck-typed so the framework is *not* a tracecheck
dependency. The user installs the framework on their side, runs their
agent, and hands the result to the adapter.
"""

from tracecheck.adapters.langgraph import langgraph_events_to_trace
from tracecheck.adapters.pydantic_ai import pydantic_ai_to_trace

__all__ = ["pydantic_ai_to_trace", "langgraph_events_to_trace"]
