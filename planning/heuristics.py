from __future__ import annotations

from planning.pddl import (
    ActionSchema,
    State,
    Objects,
    get_applicable_actions,
)


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###

    current_fluents = set(state)
    goal_fluents = set(goal)

    # Fluentes del objetivo que faltan
    missing_goals = goal_fluents - current_fluents

    # Si ya se cumple el objetivo
    if not missing_goals:
        return 0

    actions_used = 0
    remaining = set(missing_goals)

    # Greedy set-cover
    while remaining:

        best_action = None
        best_covered = set()

        # Como ignoramos precondiciones,
        # revisamos todas las acciones del dominio
        for action in domain:

            added = set(action.add_list)

            covered = remaining.intersection(added)

            if len(covered) > len(best_covered):
                best_action = action
                best_covered = covered

        # Si ninguna acción ayuda, detener
        if best_action is None or not best_covered:
            break

        # Remover objetivos cubiertos
        remaining -= best_covered

        actions_used += 1

    return actions_used

    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so hill-climbing always makes progress and cannot loop.

    Algorithm (hill-climbing on the relaxed problem):
      1. Start from the current state with a relaxed (monotone) apply function.
      2. At each step, pick the grounded action that adds the most unsatisfied
         goal fluents (greedy hill-climbing).
      3. Count steps until all goal fluents are satisfied (or until no progress).

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###

    # Estado relajado:
    # ignoramos delete lists, por lo que el estado solo crece
    current_fluents = set(state)
    goal_fluents = set(goal)

    steps = 0

    # Límite de seguridad para evitar loops infinitos
    max_steps = 100

    # Mientras el goal no esté satisfecho
    while not goal_fluents.issubset(current_fluents):

        applicable_actions = get_applicable_actions(
            State(frozenset(current_fluents)),
            domain,
            objects,
        )

        # Si no hay acciones aplicables
        if not applicable_actions:
            return 999

        best_action = None
        best_gain = -1

        missing_goals = goal_fluents - current_fluents

        # Escoger la acción que más ayude
        for action in applicable_actions:

            added = set(action.add_list)

            gain = len(added.intersection(missing_goals))

            # Preferimos la acción con mayor ganancia
            if gain > best_gain:
                best_gain = gain
                best_action = action

        # Si no mejora directamente el goal,
        # igual tomar una acción para seguir creciendo
        if best_action is None:
            best_action = applicable_actions[0]

        # Aplicación relajada:
        # solo agregamos fluentes
        current_fluents.update(best_action.add_list)

        steps += 1

        # Evitar ciclos infinitos
        if steps > max_steps:
            return 999

    return steps

    ### End of your code ###