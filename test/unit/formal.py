"""A tiny state-transition language for structural tests.

A `State` is the committed component-set snapshot of a World plus `next_id`, the id the next spawn will get.
It is a Mapping, so it compares directly to a plain dict of lists: `assert state == {0: [C1], 2: [C1, C3]}`.
That dict comparison is the structural projection only. Two States compare equal only when both the mapping and
`next_id` match: `assert state.next_id == 3`. The World remains available explicitly as `state.world` for
inspection beyond the formula (pools, id maps, entity handles). `act` applies an ordered list of actions to a
copy and commits once, returning a new State. The action constructors are the whole alphabet:

    addc(eid, C)       entity.add_component
    rmc(eid, C)        entity.remove_component
    adde([C...])       world.add_entity
    rme(eid)           world.remove_entity

There is no Action base class: the four private frozen classes are the whole alphabet, and `act` accepts their
exact types. A test cannot add a fifth verb or smuggle in an arbitrary World mutation.

`adde` does not name an id; the World allocates it, and the dict comparison checks it. Field data is only a
way to satisfy required fields here; dtype/shape behaviour has its own tests.

A rejected action raises `ActError`. `ActError.state` is the recovered state: the accepted prefix is committed
and the input State is untouched. Tests assert `exc.value.state == {...}`.
"""
from __future__ import annotations
from collections.abc import Mapping
import copy
from dataclasses import dataclass
from typing import Any, TypeAlias

from microecs import World, Component

StateSnapshot: TypeAlias = dict[int, frozenset[type[Component]]]


def _copy_world(world: World) -> World:
    """A structural deepcopy that keeps the lazy parts lazy.

    Entity views refuse to be copied; their cache entries become None and get_entity() rebuilds them against the
    copy. QueryResult objects are derived state and start empty.
    """
    memo: dict[int, Any] = {id(world._qr_cache): {}}
    for entity in world.live_entities.values():
        if entity is not None:
            memo[id(entity)] = None
    return copy.deepcopy(world, memo)


class State(Mapping[int, frozenset[type[Component]]]):
    """The committed world plus its eid -> component-set snapshot and the next id to allocate.

    Equality between two States is equality of the whole state, including `next_id`. Equality with a plain
    mapping is the structural projection only: the live entities, no allocation history. That projection is
    useful in formulas, but it is not a congruence for `adde`; two States with the same mapping can hand out
    different ids. Use `state.next_id` when the future spawn id matters.
    """

    def __init__(self, world: World):
        assert len(world._command_buffer) == 0, "State requires a committed world"
        self.world = world
        self.entities = snapshot(world)
        self.next_id = world._last_id + 1

    def __getitem__(self, eid: int) -> frozenset[type[Component]]:
        return self.entities[eid]

    def __iter__(self):
        return iter(self.entities)

    def __len__(self) -> int:
        return len(self.entities)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return self.entities == other.entities and self.next_id == other.next_id
        if isinstance(other, Mapping):
            # Expected values are component iterables (lists/sets); component order is irrelevant.
            return self.entities == {eid: frozenset(components) for eid, components in other.items()}
        return NotImplemented

    def __repr__(self) -> str:
        return f"State({self.entities}, next_id={self.next_id})"


@dataclass(frozen=True)
class _AddComponent:
    eid: int
    component: type[Component]
    data: dict[str, Any]

    def apply(self, world: World) -> None:
        world.get_entity(self.eid).add_component(self.component, **self.data)


@dataclass(frozen=True)
class _RemoveComponent:
    eid: int
    component: type[Component]

    def apply(self, world: World) -> None:
        world.get_entity(self.eid).remove_component(self.component)


@dataclass(frozen=True)
class _AddEntity:
    components: list[type[Component]]
    data: dict[str, Any]

    def apply(self, world: World) -> None:
        world.add_entity(self.components, **self.data)


@dataclass(frozen=True)
class _RemoveEntity:
    eid: int

    def apply(self, world: World) -> None:
        world.remove_entity(self.eid)


Action: TypeAlias = _AddComponent | _RemoveComponent | _AddEntity | _RemoveEntity
_VERBS = (_AddComponent, _RemoveComponent, _AddEntity, _RemoveEntity)


class ActError(Exception):
    """A rejected action, carrying the state produced by the accepted prefix.

    Tests assert `exc.value.state`; they do not inspect the buffer or the recovery mechanics.
    """

    def __init__(self, cause: BaseException, state: State):
        super().__init__(str(cause))
        self.cause = cause
        self.state = state


def setup(entities: dict[int, list[type[Component]]], components: list[type[Component]],
          data: dict[int, dict[str, Any]] | None = None) -> State:
    """Spawn `entities` with contiguous eids 0..N-1, commit once, and return the resulting state."""
    eids = sorted(entities)
    assert eids == list(range(len(eids))), f"setup eids must be contiguous from 0, got {eids}"
    world = World(components=list(components))
    for eid in sorted(entities):
        assert world.add_entity(entities[eid], **(data or {}).get(eid, {})) == eid
    world.update()
    assert_pool_ids_invariants(world)
    return State(world)


def act(state: State, actions: list[Action]) -> State:
    """Apply the actions in order and commit once, returning a new State.

    The copy is what gets returned, so the input State is never mutated. If an action raises, the accepted
    prefix is committed and carried by `ActError.state`; the input stays unchanged.
    """
    actions = list(actions)                 # one pass: validate and apply must see the same sequence
    for action in actions:
        assert type(action) in _VERBS, f"act() accepts only the four formal verbs, got {action!r}"

    world = _copy_world(state.world)
    try:
        for action in actions:
            action.apply(world)
    except Exception as cause:
        world.update()
        assert_pool_ids_invariants(world)
        raise ActError(cause, State(world)) from cause

    world.update()
    assert_pool_ids_invariants(world)
    return State(world)


def addc(eid: int, component: type[Component], **data: Any) -> Action:
    """entity.add_component."""
    return _AddComponent(eid, component, dict(data))


def rmc(eid: int, component: type[Component]) -> Action:
    """entity.remove_component."""
    return _RemoveComponent(eid, component)


def adde(components: list[type[Component]], **data: Any) -> Action:
    """world.add_entity."""
    return _AddEntity(list(components), dict(data))


def rme(eid: int) -> Action:
    """world.remove_entity."""
    return _RemoveEntity(eid)


def snapshot(world: World) -> StateSnapshot:
    """The committed state: eid -> component set."""
    return {eid: frozenset(world.get_entity(eid).get_components()) for eid in world.live_entities}


def assert_pool_ids_invariants(world: World) -> None:
    """The reverse id-map mirrors the pools exactly: no orphan/missing lists, one id per row, every id sits at
    the row it claims, and the union of all ids is precisely the live set (the command buffer is committed)."""
    assert {id(p) for p in world._pool_ids} == {id(p) for p in world.pools.values()}   # no orphan / missing
    seen = set()
    for pool, ids in world._pool_ids.items():
        assert len(ids) == len(pool)                                                   # one id per row
        for ix, eid in enumerate(ids):
            assert world._eid_to_pool_ix[eid] == (pool, ix)                            # ids[ix] sits at row ix
            seen.add(eid)
    assert seen == set(world.live_entities)                                            # exactly the live entities
