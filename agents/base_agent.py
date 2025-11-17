"""Base agent class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from utils.types import FactCheckingState
from utils.logger import log


class BaseAgent(ABC):
    """Base class for all agents in the fact-checking system."""

    def __init__(self, name: str, description: str):
        """
        Initialize the agent.

        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description
        log.info(f"Initialized agent: {name}")

    @abstractmethod
    async def process(self, state: FactCheckingState) -> FactCheckingState:
        """
        Process the state and return updated state.

        Args:
            state: Current fact-checking state

        Returns:
            Updated fact-checking state
        """
        pass

    def log_action(self, state: FactCheckingState, action: str) -> None:
        """
        Log agent action to reasoning trace.

        Args:
            state: Current state
            action: Action description
        """
        log.info(f"[{self.name}] {action}")
        state.reasoning_trace.append(f"{self.name}: {action}")

        if self.name not in state.agents_involved:
            state.agents_involved.append(self.name)

    def validate_state(self, state: FactCheckingState) -> bool:
        """
        Validate that state has required fields for this agent.

        Args:
            state: State to validate

        Returns:
            True if valid, False otherwise
        """
        if not state.original_claim:
            log.error(f"[{self.name}] State missing original_claim")
            return False
        return True

    async def __call__(self, state: FactCheckingState) -> FactCheckingState:
        """
        Make agent callable.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        if not self.validate_state(state):
            log.error(f"[{self.name}] Invalid state, skipping processing")
            return state

        try:
            return await self.process(state)
        except Exception as e:
            log.error(f"[{self.name}] Error during processing: {str(e)}")
            self.log_action(state, f"Error occurred: {str(e)}")
            return state
