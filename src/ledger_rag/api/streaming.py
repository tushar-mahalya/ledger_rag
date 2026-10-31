import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

STREAM_BUFFER_SIZE = 16
SendEvent = Callable[[str], Awaitable[None]]


async def buffered_events(produce: Callable[[SendEvent], Awaitable[None]]) -> AsyncIterator[str]:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=STREAM_BUFFER_SIZE)

    async def run() -> None:
        await produce(queue.put)

    producer = asyncio.create_task(run(), name="rag-stream-producer")
    pending: asyncio.Task[str] | None = None
    try:
        while not producer.done() or not queue.empty():
            pending = asyncio.create_task(queue.get(), name="rag-stream-event")
            await asyncio.wait((producer, pending), return_when=asyncio.FIRST_COMPLETED)
            if pending.done():
                yield pending.result()
            else:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            pending = None
        await producer
    finally:
        producer.cancel()
        tasks: list[asyncio.Task[object]] = [producer]
        if pending is not None:
            pending.cancel()
            tasks.append(pending)
        await asyncio.gather(*tasks, return_exceptions=True)
