from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic
from planning.pddl import get_all_groundings
from collections import defaultdict
from heapq import heappush, heappop
from itertools import count



# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    medical_posts = set(problem.objects["medical_posts"])

    def is_useful_for_rescue(state: State, action: Action) -> bool:
        """Avoid reversible detours that cannot help achieve rescue goals."""
        if action.name.startswith("PutDown"):
            at_effects = [fluent for fluent in action.add_list if fluent[0] == "At"]
            if not at_effects:
                return True
            _pred, obj, loc = at_effects[0]
            if obj in problem.objects["supplies"]:
                return False
            if obj in problem.objects["patients"] and loc not in medical_posts:
                return False

        if action.name.startswith("PickUp"):
            held_objects = [
                fluent[2]
                for fluent in action.add_list
                if fluent[0] == "Holding"
            ]
            if not held_objects:
                return True
            obj = held_objects[0]
            if obj in problem.objects["supplies"]:
                return not any(fluent[0] == "SuppliesReady" for fluent in state)
            if obj in problem.objects["patients"]:
                patient_locations = [
                    fluent[2]
                    for fluent in action.del_list
                    if fluent[0] == "At" and fluent[1] == obj
                ]
                return (
                    not patient_locations
                    or patient_locations[0] not in medical_posts
                )

        return True

    frontier = Queue()
    frontier.push((start, []))
    visited = {start}

    while not frontier.isEmpty():
        state, plan = frontier.pop()
        for next_state, action, _cost in problem.getSuccessors(state):
            if not is_useful_for_rescue(state, action):
                continue
            if next_state in visited:
                continue

            next_plan = plan + [action]
            if problem.isGoalState(next_state):
                return next_plan

            visited.add(next_state)
            frontier.push((next_state, next_plan))

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    goal = frozenset(goal_set)

    if action.add_list.isdisjoint(goal):
        return None

    if not action.del_list.isdisjoint(goal):
        return None

    regressed_goal = frozenset((goal - action.add_list) | action.precond_pos)

    if not action.precond_neg.isdisjoint(regressed_goal):
        return None

    return regressed_goal
    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    initial_state = frozenset(problem.initial_state)
    start_goal = frozenset(problem.goal)

    all_actions = get_all_groundings(problem.domain, problem.objects)

    static_predicates = {"Adjacent", "MedicalPost", "Pickable"}
    optimized_actions = []

    for action in all_actions:
        valid_action = True

        for fluent in action.precond_pos:
            if fluent[0] in static_predicates and fluent not in initial_state:
                valid_action = False
                break

        if valid_action:
            optimized_actions.append(action)

    all_actions = optimized_actions

    all_actions.sort(key=lambda action: 1 if action.name.startswith("Move(") else 0)

    frontier = [(start_goal, [])]
    frontier_index = 0

    visited = {start_goal}

    problem._expanded = 0

    while frontier_index < len(frontier):
        current_goal, plan = frontier[frontier_index]
        frontier_index += 1

        current_goal = frozenset(current_goal)
        problem._expanded += 1

        if len(current_goal - initial_state) == 0:
            return plan

        for action in all_actions:

            if action.add_list.isdisjoint(current_goal):
                continue

            regressed_goal = regress(current_goal, action)

            if regressed_goal is None:
                continue

            regressed_goal = frozenset(regressed_goal)

            if regressed_goal in visited:
                continue

            inconsistent = False

            positions = {}

            for fluent in regressed_goal:
                if len(fluent) == 0:
                    continue

                predicate = fluent[0]

                if predicate == "At" and len(fluent) >= 3:
                    entity = fluent[1]
                    location = fluent[2]

                    if entity in positions and positions[entity] != location:
                        inconsistent = True
                        break

                    positions[entity] = location

            if inconsistent:
                continue

            hands_free_robots = set()

            for fluent in regressed_goal:
                if len(fluent) >= 2 and fluent[0] == "HandsFree":
                    hands_free_robots.add(fluent[1])

            holdings = {}

            for fluent in regressed_goal:
                if len(fluent) >= 3 and fluent[0] == "Holding":
                    robot = fluent[1]
                    obj = fluent[2]

                    if robot in hands_free_robots:
                        inconsistent = True
                        break

                    if robot in holdings and holdings[robot] != obj:
                        inconsistent = True
                        break

                    holdings[robot] = obj

            if inconsistent:
                continue

            for fluent in regressed_goal:
                if len(fluent) > 0 and fluent[0] in static_predicates:
                    if fluent not in initial_state:
                        inconsistent = True
                        break

            if inconsistent:
                continue
    
            visited.add(regressed_goal)

            frontier.append((regressed_goal, [action] + plan))

    print("No se encontró plan")

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###
    start = problem.getStartState()

    if problem.isGoalState(start):
        return []

    frontier = PriorityQueue()

    start_h = heuristic(
        start,
        problem.goal,
        problem.domain,
        problem.objects,
    )

    frontier.push((start, [], 0), start_h)

    best_cost = {start: 0}

    problem._expanded = 0

    while not frontier.isEmpty():

        state, plan, cost_so_far = frontier.pop()

        if cost_so_far > best_cost.get(state, float("inf")):
            continue

        problem._expanded += 1

        if problem.isGoalState(state):
            return plan

        for next_state, action, step_cost in problem.getSuccessors(state):

            new_cost = cost_so_far + step_cost

            if new_cost >= best_cost.get(next_state, float("inf")):
                continue

            best_cost[next_state] = new_cost

            h_value = heuristic(
                next_state,
                problem.goal,
                problem.domain,
                problem.objects,
            )

            priority = new_cost + h_value

            new_plan = plan + [action]

            frontier.push((next_state, new_plan, new_cost), priority)

    return []
    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
