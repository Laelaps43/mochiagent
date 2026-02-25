"""
Session State Machine - 会话状态机
使用transitions库实现异步状态机
"""

from typing import Callable, Optional

from loguru import logger
from transitions.extensions.asyncio import AsyncMachine

from agent.types import SessionState


class SessionStateMachine:
    """
    会话状态机
    管理会话的状态转换
    """

    # 定义状态转换规则
    TRANSITIONS = [
        # 从IDLE可以转到PROCESSING
        {
            "trigger": "start_processing",
            "source": SessionState.IDLE.value,
            "dest": SessionState.PROCESSING.value,
        },
        # 从PROCESSING可以转到多个状态
        {
            "trigger": "start_streaming",
            "source": SessionState.PROCESSING.value,
            "dest": SessionState.STREAMING.value,
        },
        {
            "trigger": "wait_for_tool",
            "source": SessionState.PROCESSING.value,
            "dest": SessionState.WAITING_TOOL.value,
        },
        {
            "trigger": "complete",
            "source": SessionState.PROCESSING.value,
            "dest": SessionState.IDLE.value,
        },
        {
            "trigger": "fail",
            "source": SessionState.PROCESSING.value,
            "dest": SessionState.ERROR.value,
        },
        # 从STREAMING可以转到WAITING_TOOL或IDLE
        {
            "trigger": "wait_for_tool",
            "source": SessionState.STREAMING.value,
            "dest": SessionState.WAITING_TOOL.value,
        },
        {
            "trigger": "complete",
            "source": SessionState.STREAMING.value,
            "dest": SessionState.IDLE.value,
        },
        {
            "trigger": "fail",
            "source": SessionState.STREAMING.value,
            "dest": SessionState.ERROR.value,
        },
        # 从WAITING_TOOL可以转回PROCESSING或ERROR
        {
            "trigger": "continue_processing",
            "source": SessionState.WAITING_TOOL.value,
            "dest": SessionState.PROCESSING.value,
        },
        {
            "trigger": "fail",
            "source": SessionState.WAITING_TOOL.value,
            "dest": SessionState.ERROR.value,
        },
        # 从ERROR可以重置到IDLE
        {
            "trigger": "reset",
            "source": SessionState.ERROR.value,
            "dest": SessionState.IDLE.value,
        },
        # 任何状态都可以终止
        {"trigger": "terminate", "source": "*", "dest": SessionState.TERMINATED.value},
    ]

    def __init__(self, session_id: str, on_state_change: Optional[Callable] = None):
        """
        初始化状态机

        Args:
            session_id: 会话ID
            on_state_change: 状态改变时的回调函数 async def callback(session_id, from_state, to_state)
        """
        self.session_id = session_id
        self._on_state_change = on_state_change

        self.state = SessionState.IDLE.value

        # 创建异步状态机
        self.machine = AsyncMachine(
            model=self,
            states=[state.value for state in SessionState],
            transitions=self.TRANSITIONS,
            initial=SessionState.IDLE.value,
            auto_transitions=False,  # 禁用自动转换
            send_event=True,  # 传递事件对象到回调
        )

        # 添加状态转换后的回调
        self.machine.after_state_change.append(self._after_state_change)

        logger.info(f"StateMachine created for session {session_id}, initial state: {self.state}")

    async def _after_state_change(self, event_data):
        """状态转换后的回调"""
        from_state = event_data.transition.source
        to_state = event_data.transition.dest

        logger.info(f"Session {self.session_id} state changed: {from_state} -> {to_state}")

        # 调用外部回调
        if self._on_state_change:
            try:
                await self._on_state_change(self.session_id, from_state, to_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}", exc_info=True)

    @property
    def current_state(self) -> SessionState:
        """获取当前状态"""
        return SessionState(self.state)

    def can_transition(self, trigger: str) -> bool:
        """
        检查是否可以执行某个转换

        Args:
            trigger: 触发器名称 (如 'start_processing')

        Returns:
            bool: 是否可以转换
        """
        return self.machine.get_triggers(self.state) and trigger in self.machine.get_triggers(
            self.state
        )

    async def transition_to(self, new_state: SessionState) -> bool:
        """
        尝试转换到指定状态

        Args:
            new_state: 目标状态

        Returns:
            bool: 是否成功转换
        """
        current = self.current_state

        # 已经在目标状态
        if current == new_state:
            return True

        # 查找可用的触发器
        trigger_map = {
            (SessionState.IDLE, SessionState.PROCESSING): "start_processing",
            (SessionState.PROCESSING, SessionState.STREAMING): "start_streaming",
            (SessionState.PROCESSING, SessionState.WAITING_TOOL): "wait_for_tool",
            (SessionState.PROCESSING, SessionState.IDLE): "complete",
            (SessionState.PROCESSING, SessionState.ERROR): "fail",
            (SessionState.STREAMING, SessionState.WAITING_TOOL): "wait_for_tool",
            (SessionState.STREAMING, SessionState.IDLE): "complete",
            (SessionState.STREAMING, SessionState.ERROR): "fail",
            (SessionState.WAITING_TOOL, SessionState.PROCESSING): "continue_processing",
            (SessionState.WAITING_TOOL, SessionState.ERROR): "fail",
            (SessionState.ERROR, SessionState.IDLE): "reset",
        }

        trigger = trigger_map.get((current, new_state))
        if not trigger:
            logger.warning(
                f"No valid transition from {current.value} to {new_state.value} "
                f"for session {self.session_id}"
            )
            return False

        # 执行转换
        try:
            await getattr(self, trigger)()
            return True
        except Exception as e:
            logger.error(f"Failed to transition: {e}", exc_info=True)
            return False
