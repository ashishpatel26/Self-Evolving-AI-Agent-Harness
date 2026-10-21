"""OpenTelemetry spans for tasks, benchmarks, gates and evolution cycles.

Off by default (no-op). Enable with either:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   -> OTLP/HTTP (Jaeger, Tempo, Langfuse, ...)
    HARNESS_OTEL_CONSOLE=1                              -> print spans to the console
"""

import atexit
import os
from contextlib import contextmanager

from opentelemetry import trace

_tracer = trace.NoOpTracer()

if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("HARNESS_OTEL_CONSOLE"):
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()  # reads OTEL_EXPORTER_OTLP_* env vars
    else:
        exporter = ConsoleSpanExporter()
    # Own provider (not the global one) so CrewAI's internal telemetry can't swap it out.
    _provider = TracerProvider(resource=Resource.create({"service.name": "self-evolving-agent-harness"}))
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    atexit.register(_provider.shutdown)
    _tracer = _provider.get_tracer("self_evolving_ai_agent_harness")


@contextmanager
def span(name: str, **attributes):
    """`with span("harness.task", query=q) as s: s.set_attribute("success", True)`"""
    with _tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            s.set_attribute(key, value if isinstance(value, (str, bool, int, float)) else str(value))
        yield s
