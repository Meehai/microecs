from __future__ import annotations
from .utils import Command, CommandType, EntityId
from .component import ComponentType

class CommandBuffer:
    """A data structure that holds all the uncommited commands between two world updates. Support eager exceptions
       on things like adding the same component twice on the same entity"""
    def __init__(self, world: "World"): # noqa
        self.data: list[Command] = []
        self.world = world

    def clear(self):
        """Clears the buffer"""
        self.data.clear()

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

    def _get_components_state(self, component: ComponentType, existing_components: list[ComponentType],
                              entity_id: EntityId) -> int:
        # Look for the latest staste of this entity w.r.t this component given the unstaged command buffer.
        # We look in the buffer from right to left and return 1 if the component was added, -1 if it was not
        # If the component is not in the buffer at all, we check if it is already in the entity and return +1/-1 as well
        for old_command in reversed(self.data):
            if old_command.entity_id != entity_id:
                continue
            if old_command.command_type == CommandType.ADD_COMPONENT:
                old_component = old_command.args["component"]
                if component == old_component:
                    return 1
            if old_command.command_type == CommandType.REMOVE_COMPONENT:
                old_component = old_command.args
                if component == old_component:
                    return -1
        return -1 if component not in existing_components else 1

    def append(self, command: Command):
        """Appends a command to the buffer"""
        world = self.world
        entity_id = command.entity_id
        if entity_id not in world.live_entities:
            raise ValueError(f"Entity: {entity_id} not in live entities ({command})")

        if command.command_type == CommandType.ADD_ENTITY:
            fk = {k: v for k, v in command.args.items() if k != "components"}
            world._validate_components(command.args["components"], **fk)
            command.args.update(world._defaults_for(command.args["components"], **fk))

        elif command.command_type == CommandType.ADD_COMPONENT:
            component = command.args["component"]
            fk = {k: v for k, v in command.args.items() if k != "component"}
            world._validate_components([component], **fk)

            components = self._get_entity_components(entity_id)
            assert len(components) > 0, f"guaranteed to be >0 {entity_id} {components}"
            state = self._get_components_state(component, existing_components=components, entity_id=entity_id)
            if state == 1:
                raise ValueError(f"Component: {component} either added twice or exists already (id: {entity_id})")

        elif command.command_type == CommandType.REMOVE_COMPONENT:
            component = command.args
            if component not in world.component_types:
                raise ValueError(f"Unknown component: {component} not in world components {world.component_types}")

            components = self._get_entity_components(entity_id)
            assert len(components) > 0, f"guaranteed to be >0 {entity_id} {components}"
            state = self._get_components_state(component, existing_components=components, entity_id=entity_id)
            if state == -1:
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
