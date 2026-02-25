"""消息总线 - 异步事件发布订阅"""

import asyncio
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from loguru import logger

from agent.types import Event, EventType
from agent.constants import MESSAGE_BUS_QUEUE_TIMEOUT


class MessageBus:
    """异步消息总线 - 支持事件订阅、发布和外部监听"""

    def __init__(self, max_concurrent: int = 50):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"MessageBus initialized with max_concurrent={max_concurrent}")

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type.value}: {handler.__name__}")

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed from {event_type.value}: {handler.__name__}")

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)
        logger.debug(f"Published event: {event.type.value} for session {event.session_id}")

    async def _process_events(self) -> None:
        logger.info("Event processing loop started")
        while self._running:
            try:
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=MESSAGE_BUS_QUEUE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    continue
                await self._handle_event(event)
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)
        logger.info("Event processing loop stopped")

    async def _handle_event(self, event: Event) -> None:
        async with self._semaphore:
            handlers = self._subscribers.get(event.type, [])
            if handlers:
                tasks = [handler(event) for handler in handlers]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Handler {handlers[i].__name__} failed for event "
                            f"{event.type.value}: {result}"
                        )

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
