# pylint: disable=protected-access
"""command_buffer.py - data structure that eagerly manages the staging area (add/rm entity/cmpt) before world.upate()"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .utils import EntityId
from .component import ComponentType

if TYPE_CHECKING:
    from microecs import World

class CommandType(StrEnum):
    """The types of commands in the command pattern below"""
    ADD_ENTITY       = "add_entity"
    REMOVE_ENTITY    = "remove_entity"
    ADD_COMPONENT    = "add_component"
    REMOVE_COMPONENT = "remove_component"

@dataclass
class Command:
    """A command that can happen between two world.updates(), e.g. add/rm entity or components"""
    command_type: CommandType                            # The type of command in the command buffer
    entity_id: EntityId                                  # The entity this command applies to
    args: dict[str, Any] = field(default_factory=dict)   # The arguments of this command. Each command has its own shape

class CommandBuffer:
    """A data structure that holds all the uncommited commands between two world updates. Support eager exceptions
       on things like adding the same component twice on the same entity"""
    def __init__(self, world: "World"): # noqa
        self.data: list[Command] = [] # should we make this dict[EntityId, list[Command]] ?
        self.world = world
        self.removed_this_tick: set[EntityId] = set()

    def clear(self):
        """Clears the buffer"""
        self.data.clear()
        self.removed_this_tick.clear()

    def _get_entity_components(self, entity_id: EntityId) -> list[ComponentType]:
        """
        Return the entity's component set before this tick's component commands.
        The buffered components (via ADD/REMOVE_COMPONENT) are not visible here.
        Moreover, ADD_ENTITY->REMOVE_ENTITY->ADD_ENTITY cannot happen, as we always increment the id on adds.
        Finally, we for-loop forward, so we get the first (and only) staged ADD_ENTITY, making it ok.
        """
        if entity_id not in self.world._eid_to_pool_ix: # staged entity (this tick) via ADD_ENTITY
            for cmd in self.data:
                if cmd.entity_id == entity_id and cmd.command_type == CommandType.ADD_ENTITY:
                    return cmd.args["components"]
            assert False, "Invariant: live entity has a pool or a staged ADD_ENTITY"
        pool, _ = self.world._eid_to_pool_ix[entity_id]
        return self.world.pool_to_components[pool]

    def _get_component_in_buffer_state(self, entity_id: EntityId, component: ComponentType) -> bool | None:
        # Look for the latest state of this entity w.r.t this component given the unstaged command buffer.
        # If the component is not in the buffer at all, return None. Return True if the last command is an ADD_COMPONENT
        # Return False if the last command is an REMOVE_COMPONENT. Tri-state used to determine what we do with a new
        # component called twice in the same tick before world.update()
        for old_command in reversed(self.data):
            if old_command.entity_id == entity_id and old_command.args.get("component", None) == component:
                return old_command.command_type == CommandType.ADD_COMPONENT # true if ADD, false if REMOVE
        return None

    def append(self, command: Command):
        """Appends a command to the buffer"""
        world = self.world
        if (eid := command.entity_id) not in world.live_entities:
            raise ValueError(f"Entity: {eid} not in live entities ({command})")

        if command.command_type == CommandType.ADD_ENTITY:
            # nothing to do here: world.add_entity already ensures validated args (& defaults) come here.
            pass

        elif command.command_type == CommandType.REMOVE_ENTITY:
            # needed so we can fast check in world.remove_enitity if this is a no-op (same tick) or error (stale eid).
            self.removed_this_tick.add(eid)

        elif command.command_type == CommandType.ADD_COMPONENT:
            component = command.args["component"]
            fk = {k: v for k, v in command.args.items() if k != "component"}
            # same data conversion as add_entity (lists/tuples/scalars, None semantics, extras preserved)
            fk = world._convert_components_data([component], **fk)
            world._validate_component(component, strict=True, check_extra=True, **fk)

            component_state = self._get_component_in_buffer_state(eid, component)
            if component_state is True:
                raise ValueError(f"Component: {component} already was already buffered this tick (id: {eid}). "
                                 f"Run world.update() or you may have a bug where 2 systems write without a lock in "
                                  "the same frame which may cause data loss on entities.")

            if component_state is None:
                components = self._get_entity_components(eid)
                if component in components:
                    raise ValueError(f"Component: {component} already exists already, use e.set_data(). (id: {eid})")

            # If we reach here either the component is not in the entity or we flip-flop in this frame (REMOVE->ADD)
            command.args = {"component": component, **fk}

        elif command.command_type == CommandType.REMOVE_COMPONENT:
            component = command.args["component"]
            if component not in world.component_types:
                raise ValueError(f"Unknown component: {component} not in world components {world.component_types}")

            components = self._get_entity_components(eid)
            component_state = self._get_component_in_buffer_state(eid, component)
            if component_state is None: # not in buffer -> we check if the entity has the component at all
                if component not in components:
                    raise ValueError(f"{component=} is not part of the entity. Use e.add_component(). (id: {eid})")
            else: # Component is buffered. Thus we need to make sure it's flip flopped (e.g. remove after add)
                if component_state is False: # remove after remove is no-op
                    return

        # We copy here because there's no guarantee the underlying arrays won't mutate (e.g. arr.resize) before commit.
        command.args = {name: value.copy() if isinstance(value, np.ndarray) else value
                        for name, value in command.args.items()}
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
