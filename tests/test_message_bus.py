from __future__ import annotations

from agent.core.bus import MessageBus
from agent.types import Event, EventType


def _make_event(session_id: str = "sess_1") -> Event:
    return Event(type=EventType.MESSAGE_RECEIVED, session_id=session_id, data={})


class TestMessageBusSubscribeUnsubscribe:
    def test_subscribe_and_unsubscribe(self):
        bus = MessageBus(max_concurrent=4)

        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.MESSAGE_RECEIVED, handler)
        bus.unsubscribe(EventType.MESSAGE_RECEIVED, handler)

    def test_unsubscribe_nonexistent_no_error(self):
        bus = MessageBus(max_concurrent=4)

        async def handler(_event: Event) -> None:
            pass

        bus.unsubscribe(EventType.MESSAGE_RECEIVED, handler)


class TestMessageBusPublishAndProcess:
    async def test_publish_and_receive(self):
        bus = MessageBus(max_concurrent=4)
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.MESSAGE_RECEIVED, handler)
        await bus.start()

        event = _make_event()
        await bus.publish(event)
        await bus.wait_empty()
        await bus.stop()

        assert len(received) == 1
        assert received[0].type == EventType.MESSAGE_RECEIVED

    async def test_start_twice_no_duplicate(self):
        bus = MessageBus(max_concurrent=4)
        await bus.start()
        await bus.start()
        await bus.stop()

    async def test_stop_when_not_running(self):
        bus = MessageBus(max_concurrent=4)
        await bus.stop()

    async def test_multiple_subscribers(self):
        bus = MessageBus(max_concurrent=4)
        results: list[str] = []

        async def h1(_event: Event) -> None:
            results.append("h1")

        async def h2(_event: Event) -> None:
            results.append("h2")

        bus.subscribe(EventType.MESSAGE_RECEIVED, h1)
        bus.subscribe(EventType.MESSAGE_RECEIVED, h2)

        await bus.start()
        await bus.publish(_make_event())
        await bus.wait_empty()
        await bus.stop()

        assert "h1" in results
        assert "h2" in results

    async def test_handler_exception_does_not_crash_bus(self):
        bus = MessageBus(max_concurrent=4)

        async def bad_handler(_event: Event) -> None:
            raise RuntimeError("handler failure")

        bus.subscribe(EventType.MESSAGE_RECEIVED, bad_handler)

        await bus.start()
        await bus.publish(_make_event())
        await bus.wait_empty()
        await bus.stop()

    async def test_publish_multiple_events(self):
        bus = MessageBus(max_concurrent=4)
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.MESSAGE_RECEIVED, handler)
        await bus.start()

        for i in range(5):
            await bus.publish(_make_event(f"sess_{i}"))

        await bus.wait_empty()
        await bus.stop()

        assert len(received) == 5

    async def test_no_subscribers_event_discarded(self):
        bus = MessageBus(max_concurrent=4)
        await bus.start()
        await bus.publish(_make_event())
        await bus.wait_empty()
        await bus.stop()
