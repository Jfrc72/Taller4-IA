from __future__ import annotations

from collections import deque
from itertools import permutations

from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    ### Your code here ###
    if not hlas:
        return []

    def plan_signature(plan: list[Action | HLA]) -> tuple[str, ...]:
        return tuple(step.name for step in plan)

    def first_non_primitive_index(plan: list[Action | HLA], start: int = 0) -> int | None:
        for i in range(start, len(plan)):
            if not is_primitive(plan[i]):
                return i
        return None

    def execute_primitive_prefix(
        plan: list[Action | HLA],
    ) -> tuple[bool, int, frozenset]:
        state = problem.initial_state
        for i, step in enumerate(plan):
            if not is_primitive(step):
                return True, i, state
            if not is_applicable(state, step):
                return False, i, state
            state = apply_action(state, step)
        return True, len(plan), state

    robot = problem.objects["robots"][0] if problem.objects.get("robots") else None

    def refinement_can_start(
        refinement: list[Action | HLA],
        state: frozenset,
    ) -> bool:
        if not refinement:
            return True

        first_step = refinement[0]
        if is_primitive(first_step):
            return is_applicable(state, first_step)

        if getattr(first_step, "kind", None) == "Navigate":
            start_cell = getattr(first_step, "start_cell", None)
            return robot is None or ("At", robot, start_cell) in state

        return True

    def state_refinements(hla: HLA, state: frozenset) -> list[list[Action | HLA]]:
        kind = getattr(hla, "kind", None)

        if kind == "FullRescueMission":
            medical_post = getattr(hla, "medical_post")
            supplies_ready = ("SuppliesReady", medical_post) in state
            return [hla.refinements[1 if supplies_ready else 0]]

        if kind == "RescuePatient":
            ready_posts = {
                fluent[1]
                for fluent in state
                if fluent[0] == "SuppliesReady"
            }
            if ready_posts:
                ready_refinements = [
                    refinement
                    for refinement in hla.refinements
                    if refinement
                    and getattr(refinement[0], "medical_post", None) in ready_posts
                ]
                if ready_refinements:
                    return ready_refinements

        return hla.refinements

    frontier = deque([(list(hlas), 0)])
    visited = {plan_signature(list(hlas))}
    problem._expanded = 0

    while frontier:
        plan, start_index = frontier.popleft()
        problem._expanded += 1

        prefix_is_valid, checked_until, prefix_state = execute_primitive_prefix(plan)
        if not prefix_is_valid:
            continue

        hla_index = first_non_primitive_index(plan, max(start_index, checked_until))
        if hla_index is None:
            if problem.isGoalState(frozenset(prefix_state)):
                return plan  # type: ignore[return-value]
            continue

        hla = plan[hla_index]
        for refinement in state_refinements(hla, prefix_state):
            if not refinement_can_start(refinement, prefix_state):
                continue
            refined_plan = (
                plan[:hla_index] + list(refinement) + plan[hla_index + 1 :]
            )
            signature = plan_signature(refined_plan)
            if signature in visited:
                continue
            visited.add(signature)
            frontier.append((refined_plan, hla_index))

    ### End of your code ###
    return []


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###
    robot = problem.objects["robots"][0]
    cells = problem.objects["cells"]
    supplies = problem.objects["supplies"]
    patients = [
        patient
        for patient in problem.objects["patients"]
        if ("Rescued", patient) in problem.goal
    ]
    medical_posts = problem.objects["medical_posts"]

    if not patients:
        return [HLA("FullRescueProgram", [[]])]
    if not supplies or not medical_posts:
        return []

    schema_by_name = {schema.name: schema for schema in problem.domain}

    def ground(schema_name: str, **binding: object) -> Action:
        return schema_by_name[schema_name].ground(binding)

    def locations_for(predicate: str) -> dict[str, tuple[int, int]]:
        locations: dict[str, tuple[int, int]] = {}
        for fluent in problem.initial_state:
            if fluent[0] == "At" and fluent[1] in problem.objects[predicate]:
                locations[fluent[1]] = fluent[2]
        return locations

    supply_locations = locations_for("supplies")
    patient_locations = locations_for("patients")

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in cells}
    for fluent in problem.initial_state:
        if fluent[0] == "Adjacent":
            adjacency.setdefault(fluent[1], []).append(fluent[2])
    for neighbors in adjacency.values():
        neighbors.sort()

    def shortest_paths(
        start: tuple[int, int],
        goal: tuple[int, int],
        max_paths: int = 2,
    ) -> list[list[tuple[int, int]]]:
        if start == goal:
            return [[start]]

        paths: list[list[tuple[int, int]]] = []
        frontier = deque([[start]])
        shortest_length: int | None = None

        while frontier and len(paths) < max_paths:
            path = frontier.popleft()
            current = path[-1]

            if shortest_length is not None and len(path) >= shortest_length:
                continue

            for next_cell in adjacency.get(current, []):
                if next_cell in path:
                    continue

                next_path = path + [next_cell]
                if next_cell == goal:
                    shortest_length = len(next_path)
                    paths.append(next_path)
                elif shortest_length is None or len(next_path) < shortest_length:
                    frontier.append(next_path)

        return paths

    navigate_cache: dict[tuple[tuple[int, int], tuple[int, int]], HLA] = {}

    def navigate(start: tuple[int, int], goal: tuple[int, int]) -> HLA:
        key = (start, goal)
        if key not in navigate_cache:
            refinements: list[list[Action]] = []
            for path in shortest_paths(start, goal):
                moves = [
                    ground(
                        "Move",
                        r=robot,
                        from_cell=from_cell,
                        to_cell=to_cell,
                    )
                    for from_cell, to_cell in zip(path, path[1:])
                ]
                refinements.append(moves)
            hla = HLA(f"Navigate({start}->{goal})", refinements)
            hla.kind = "Navigate"
            hla.start_cell = start
            hla.goal_cell = goal
            navigate_cache[key] = hla
        return navigate_cache[key]

    prepare_cache: dict[tuple[str, tuple[int, int]], HLA] = {}

    def prepare_supplies(supply: str, medical_post: tuple[int, int]) -> HLA:
        key = (supply, medical_post)
        if key not in prepare_cache:
            supply_location = supply_locations[supply]
            refinements: list[list[Action | HLA]] = []
            for start in cells:
                refinements.append(
                    [
                        navigate(start, supply_location),
                        ground("PickUp", r=robot, obj=supply, loc=supply_location),
                        navigate(supply_location, medical_post),
                        ground(
                            "SetupSupplies",
                            r=robot,
                            s=supply,
                            loc=medical_post,
                        ),
                    ]
                )
            prepare_cache[key] = HLA(
                f"PrepareSupplies({supply},{medical_post})",
                refinements,
            )
        return prepare_cache[key]

    extract_cache: dict[tuple[str, tuple[int, int]], HLA] = {}

    def extract_patient(patient: str, medical_post: tuple[int, int]) -> HLA:
        key = (patient, medical_post)
        if key not in extract_cache:
            patient_location = patient_locations[patient]
            refinements: list[list[Action | HLA]] = []
            for start in cells:
                refinements.append(
                    [
                        navigate(start, patient_location),
                        ground("PickUp", r=robot, obj=patient, loc=patient_location),
                        navigate(patient_location, medical_post),
                        ground("PutDown", r=robot, obj=patient, loc=medical_post),
                    ]
                )
            extract_cache[key] = HLA(
                f"ExtractPatient({patient},{medical_post})",
                refinements,
            )
        return extract_cache[key]

    mission_cache: dict[tuple[str, str, tuple[int, int]], HLA] = {}

    def full_rescue_mission(
        supply: str,
        patient: str,
        medical_post: tuple[int, int],
    ) -> HLA:
        key = (supply, patient, medical_post)
        if key not in mission_cache:
            rescue = ground("Rescue", r=robot, p=patient, loc=medical_post)
            hla = HLA(
                f"FullRescueMission({supply},{patient},{medical_post})",
                [
                    [
                        prepare_supplies(supply, medical_post),
                        extract_patient(patient, medical_post),
                        rescue,
                    ],
                    [
                        extract_patient(patient, medical_post),
                        rescue,
                    ],
                ],
            )
            hla.kind = "FullRescueMission"
            hla.supply = supply
            hla.patient = patient
            hla.medical_post = medical_post
            mission_cache[key] = hla
        return mission_cache[key]

    patient_task_cache: dict[str, HLA] = {}

    def patient_rescue_task(patient: str) -> HLA:
        if patient not in patient_task_cache:
            refinements: list[list[HLA]] = []
            for medical_post in medical_posts:
                for supply in supplies:
                    refinements.append(
                        [full_rescue_mission(supply, patient, medical_post)]
                    )
            hla = HLA(
                f"RescuePatient({patient})",
                refinements,
            )
            hla.kind = "RescuePatient"
            hla.patient = patient
            patient_task_cache[patient] = hla
        return patient_task_cache[patient]

    root_refinements: list[list[HLA]] = [
        [patient_rescue_task(patient) for patient in order]
        for order in permutations(patients)
    ]

    return [HLA("FullRescueProgram", root_refinements)]

    ### End of your code ###
