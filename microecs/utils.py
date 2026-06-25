"""utils.py - common utilities used by microecs. Mostly logger and types used across the code."""
from enum import StrEnum
from dataclasses import dataclass
from typing import Any
from loggez import make_logger

PoolKey = int
EntityId = int
Shape = tuple[int, ...]

logger = make_logger("MICROECS")

class CommandType(StrEnum):
    """the types of commands in the command pattern below"""
    ADD_ENTITY       = "add_entity"
    REMOVE_ENTITY    = "remove_entity"
    ADD_COMPONENT    = "add_component"
    REMOVE_COMPONENT = "remove_component"

@dataclass
class Command:
    """class for the list of commands that can happen between two world.updates(), e.g. add/rmv entity or components"""
    command_type: CommandType
    entity_id: EntityId
    args: Any | None = None
