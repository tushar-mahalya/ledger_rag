from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

tracer = trace.get_tracer(__name__)


@contextmanager
def safe_span(name: str) -> Iterator[Span]:
    # Upstream exception messages can contain documents, prompts or credentials.
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            yield span
        except BaseException:
            span.set_status(Status(StatusCode.ERROR))
            raise
