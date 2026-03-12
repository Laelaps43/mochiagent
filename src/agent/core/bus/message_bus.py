"""消息总线 - 异步事件发布订阅"""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable

from loguru import logger

from agent.config.system import MessageBusConfig
from agent.types import Event, EventType


class MessageBus:
    """异步消息总线 - 支持事件订阅、发布和外部监听"""

    def __init__(self, max_concurrent: int = 50):
        self._subscribers: dict[EventType, list[Callable[["Event"], Awaitable[None]]]] = (
            defaultdict(list)
        )
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running: bool = False
        self._process_task: asyncio.Task[None] | None = None
        self._max_concurrent: int = max_concurrent
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"MessageBus initialized with max_concurrent={max_concurrent}")

    def subscribe(
        self, event_type: EventType, handler: Callable[["Event"], Awaitable[None]]
    ) -> None:
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type.value}: {handler.__name__}")

    def unsubscribe(
        self, event_type: EventType, handler: Callable[["Event"], Awaitable[None]]
    ) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed from {event_type.value}: {handler.__name__}")

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)
        logger.debug(f"Published event: {event.type.value} for session {event.session_id}")

    async def _process_events(self) -> None:
        logger.info("Event processing loop started")
        queue_timeout = MessageBusConfig().queue_timeout
        pending: set[asyncio.Task[None]] = set()
        while self._running or not self._queue.empty():
            try:
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=queue_timeout)
                except asyncio.TimeoutError:
                    if not self._running:
                        break
                    continue
                _ = await self._semaphore.acquire()
                task = asyncio.create_task(self._handle_event(event))

                def _on_done(_task: asyncio.Task[None]) -> None:
                    pending.discard(_task)
                    self._queue.task_done()

                pending.add(task)
                task.add_done_callback(_on_done)
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)
        # 等待所有已派发的 handler 完成
        if pending:
            _ = await asyncio.gather(*pending, return_exceptions=True)
        logger.info("Event processing loop stopped")

    async def _handle_event(self, event: Event) -> None:
        try:
            handlers = list(self._subscribers.get(event.type, []))
            if handlers:
                tasks = [handler(event) for handler in handlers]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Handler {handlers[i].__name__} failed for event {event.type.value}: {result}"
                        )
        finally:
            self._semaphore.release()

    async def start(self) -> None:
        if self._running:
            logger.warning("MessageBus already running")
            return
        self._running = True
        self._process_task = asyncio.create_task(self._process_events())
        logger.info("MessageBus started")

    async def stop(self) -> None:
        if not self._running:
            return
        logger.info("Stopping MessageBus...")
        self._running = False
        if self._process_task:
            await self._process_task
        logger.info("MessageBus stopped")

    async def wait_empty(self) -> None:
        await self._queue.join()
