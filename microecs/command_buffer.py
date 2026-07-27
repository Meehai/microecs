# pylint: disable=protected-access
"""command_buffer.py - data structure that eagerly manages the staging area (add/rm entity/cmpt) before world.upate()"""
from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from enum import StrEnum

from .utils import EntityId
from .component import ComponentType

class CommandType(StrEnum):
    """The types of commands in the command pattern below"""
    ADD_ENTITY       = "add_entity"
    REMOVE_ENTITY    = "remove_entity"
    ADD_COMPONENT    = "add_component"
    REMOVE_COMPONENT = "remove_component"

@dataclass
class Command:
    """A command that can happen between two world.updates(), e.g. add/rm entity or components"""
    command_type: CommandType
    entity_id: EntityId
    args: Any | None = None

class CommandBuffer:
    """A data structure that holds all the uncommited commands between two world updates. Support eager exceptions
       on things like adding the same component twice on the same entity"""
    def __init__(self, world: "World"): # noqa
        self.data: list[Command] = []
        self.world = world
        self.removed_this_tick: set[EntityId] = set()

    def clear(self):
        """Clears the buffer"""
        self.data.clear()
        self.removed_this_tick.clear()

    def _get_entity_components(self, entity_id: EntityId) -> list[ComponentType]: # noqa
        # This is the case for uncommited entities
        if entity_id not in self.world._eid_to_pool_ix:
            # uncommitted spawn: base = the components it was spawned with
            for cmd in self.data:
                if cmd.entity_id == entity_id and cmd.command_type == CommandType.ADD_ENTITY:
                    return cmd.args["components"]
            return [] # Entity should exist so this shouldn't be reached technically. We have an assert at call site.
        pool, _ = self.world._eid_to_pool_ix[entity_id]
        return self.world.pool_to_components[pool]

    def _entity_has_buffered_component(self, entity_id: EntityId, component: ComponentType,
                                       existing_components: list[ComponentType]) -> bool:
        # Look for the latest state of this entity w.r.t this component given the unstaged command buffer.
        # We look in the buffer from right to left and return True if the component was added, False otherwise.
        # If the component is not in the buffer at all, we check if it is already in the entity. Returns a bool.
        for old_command in reversed(self.data):
            if old_command.entity_id != entity_id:
                continue
            if old_command.command_type == CommandType.ADD_COMPONENT:
                old_component = old_command.args["component"]
                if component == old_component:
                    return True
            if old_command.command_type == CommandType.REMOVE_COMPONENT:
                old_component = old_command.args
                if component == old_component:
                    return False
        return component in existing_components

    def append(self, command: Command):
        """Appends a command to the buffer"""
        world = self.world
        entity_id = command.entity_id
        if entity_id not in world.live_entities:
            raise ValueError(f"Entity: {entity_id} not in live entities ({command})")

        if command.command_type == CommandType.ADD_ENTITY:
            # nothing to do here: world.add_entity already ensures validated args (& defaults) come here.
            pass

        elif command.command_type == CommandType.REMOVE_ENTITY:
            # needed so we can fast check in world.remove_enitity if this is a no-op (same tick) or error (stale eid).
            self.removed_this_tick.add(command.entity_id)

        elif command.command_type == CommandType.ADD_COMPONENT:
            component = command.args["component"]
            fk = {k: v for k, v in command.args.items() if k != "component"}
            world._validate_component(component, strict=True, check_extra=True, **fk)

            components = self._get_entity_components(entity_id)
            assert len(components) > 0, f"guaranteed to be >0 {entity_id} {components}"
            has_component = self._entity_has_buffered_component(entity_id, component, existing_components=components)
            if has_component:
                raise ValueError(f"Component: {component} either added twice or exists already (id: {entity_id})")

        elif command.command_type == CommandType.REMOVE_COMPONENT:
            component = command.args # TODO: use command.args["component"] for consistency
            if component not in world.component_types:
                raise ValueError(f"Unknown component: {component} not in world components {world.component_types}")

            components = self._get_entity_components(entity_id)
            assert len(components) > 0, f"guaranteed to be >0 {entity_id} {components}"
            has_component = self._entity_has_buffered_component(entity_id, component, existing_components=components)
            if not has_component:
                raise ValueError(f"Component: {component} either removed twice or doesn't exist (id: {entity_id})")

        self.data.append(command)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other: CommandBuffer | list[Command]):
        if isinstance(other, list):
            return self.data == other
        elif isinstance(other, CommandBuffer):
            return self.data == other.data
        else:
            return NotImplemented
