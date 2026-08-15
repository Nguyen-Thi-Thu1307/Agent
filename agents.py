"""Two-cell vacuum world and the seven agent types from Russell & Norvig, ch. 2.

Performance measure: +1 per clean cell per step, -move_cost per move.
"""
from collections import deque

CLEAN, DIRTY, UNKNOWN = "clean", "dirty", "unknown"
CELLS = ("A", "B")
SUCK, LEFT, RIGHT, NOOP = "Suck", "Left", "Right", "NoOp"

REGROW = {"A": 0.04, "B": 0.20}


class Rng:
    """Small LCG so the Python and browser versions produce the same runs."""

    def __init__(self, seed=0):
        self.s = (seed * 2654435761 + 12345) & 0xFFFFFFFF

    def random(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s / 4294967296.0

    def randrange(self, n):
        return int(self.random() * n)

    def choice(self, seq):
        return seq[self.randrange(len(seq))]


def move_from(cell):
    return RIGHT if cell == "A" else LEFT


class World:
    def __init__(self, dynamic=False, seed=0, move_cost=1):
        self.dirt = {"A": True, "B": True}
        self.pos = "A"
        self.score = 0
        self.moves = 0
        self.dynamic = dynamic
        self.move_cost = move_cost
        self.rng = Rng(seed)

    def percept(self):
        return self.pos, DIRTY if self.dirt[self.pos] else CLEAN

    def step(self, action):
        if action == SUCK:
            self.dirt[self.pos] = False
        elif action in (LEFT, RIGHT):
            self.pos = "B" if self.pos == "A" else "A"
            self.moves += 1

        self.score += sum(1 for c in CELLS if not self.dirt[c])
        if action in (LEFT, RIGHT):
            self.score -= self.move_cost

        if self.dynamic:
            for c in CELLS:
                if self.rng.random() < REGROW[c]:
                    self.dirt[c] = True


class TableDriven:
    name = "Table-driven"

    TABLE = {
        (("A", CLEAN),): RIGHT,
        (("A", DIRTY),): SUCK,
        (("B", CLEAN),): LEFT,
        (("B", DIRTY),): SUCK,
        (("A", CLEAN), ("A", CLEAN)): RIGHT,
        (("A", CLEAN), ("A", DIRTY)): SUCK,
    }

    def __init__(self, **_):
        self.history = []
        self.detail = {}

    def __call__(self, percept):
        self.history.append(percept)
        key = tuple(self.history)
        while key and key not in self.TABLE:
            key = key[1:]
        action = self.TABLE.get(key, NOOP)
        self.detail = {"matched": key if key in self.TABLE else None,
                       "seen": len(self.history), "action": action}
        return action

    def state(self):
        return "percept sequence length %d" % len(self.history)


class SimpleReflex:
    name = "Simple reflex"

    def __init__(self, **_):
        pass

    def __init__(self, **_):
        self.detail = {}

    def __call__(self, percept):
        cell, status = percept
        action = SUCK if status == DIRTY else move_from(cell)
        self.detail = {"status": status, "action": action}
        return action

    def state(self):
        return "no internal state"


class ModelBasedReflex:
    name = "Model-based reflex"

    def __init__(self, **_):
        self.belief = {"A": UNKNOWN, "B": UNKNOWN}
        self.detail = {}

    def __call__(self, percept):
        cell, status = percept
        self.belief[cell] = status
        if status == DIRTY:
            self.belief[cell] = CLEAN
            action, why = SUCK, "dirty"
        elif all(v == CLEAN for v in self.belief.values()):
            action, why = NOOP, "both known clean"
        else:
            action, why = move_from(cell), "other cell unknown"
        self.detail = {"why": why, "action": action, "belief": dict(self.belief)}
        return action

    def state(self):
        return "belief: " + ", ".join("%s=%s" % kv for kv in self.belief.items())


class GoalBased:
    """Explicit goal plus breadth-first search over the belief space.

    Changing the goal changes behaviour without touching any rule.
    """

    name = "Goal-based"

    def __init__(self, goal="both clean", **_):
        self.goal = goal
        self.belief = {"A": UNKNOWN, "B": UNKNOWN}
        self.plan = []
        self.expanded = 0
        self.detail = {}

    def _satisfied(self, belief):
        if self.goal == "A clean":
            return belief["A"] == CLEAN
        return belief["A"] == CLEAN and belief["B"] == CLEAN

    def _search(self, cell):
        # unknown counts as dirty: pessimistic but safe
        start = (cell, self.belief["A"] == CLEAN, self.belief["B"] == CLEAN)
        queue = deque([(start, [])])
        seen = {start}
        self.expanded = 0
        while queue:
            (pos, a_clean, b_clean), path = queue.popleft()
            self.expanded += 1
            if self._satisfied({"A": CLEAN if a_clean else DIRTY,
                                "B": CLEAN if b_clean else DIRTY}):
                return path
            if len(path) > 6:
                continue
            suck = (pos, True, b_clean) if pos == "A" else (pos, a_clean, True)
            move = ("B" if pos == "A" else "A", a_clean, b_clean)
            for nxt, action in ((suck, SUCK), (move, move_from(pos))):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [action]))
        return []

    def __call__(self, percept):
        cell, status = percept
        self.belief[cell] = status
        if self._satisfied(self.belief):
            self.plan = []
            self.detail = {"reached": True, "action": NOOP, "belief": dict(self.belief)}
            return NOOP
        self.plan = self._search(cell)
        if not self.plan:
            self.detail = {"reached": False, "action": NOOP, "belief": dict(self.belief),
                           "plan": [], "expanded": self.expanded}
            return NOOP
        action = self.plan[0]
        if action == SUCK:
            self.belief[cell] = CLEAN
        self.detail = {"reached": False, "action": action, "belief": dict(self.belief),
                       "plan": list(self.plan), "expanded": self.expanded}
        return action

    def state(self):
        plan = " -> ".join(self.plan) if self.plan else "goal reached"
        return "goal: %s | plan: %s" % (self.goal, plan)


class UtilityBased:
    """Weighs expected gain against move cost.

    Belief decays: a cell cleaned k steps ago is dirty again with probability
    1 - (1 - r)^k, where r is an assumed regrowth rate, the same for every cell.
    """

    name = "Utility-based"

    def __init__(self, move_cost=1, horizon=10, assumed_rate=0.10, **_):
        self.detail = {}
        self.move_cost = move_cost
        self.horizon = horizon
        self.rate = assumed_rate
        self.belief = {"A": UNKNOWN, "B": UNKNOWN}
        self.away = {"A": 0, "B": 0}
        self.note = ""

    def _p_dirty(self, cell):
        if self.belief[cell] == UNKNOWN:
            return 1.0
        return 1.0 - (1.0 - self.rate) ** self.away[cell]

    def __call__(self, percept):
        cell, status = percept
        for c in CELLS:
            self.away[c] += 1
        self.away[cell] = 0
        self.belief[cell] = status

        if status == DIRTY:
            self.belief[cell] = CLEAN
            self.note = "current cell is dirty, sucking costs nothing"
            self.detail = {"dirty_here": True, "action": SUCK, "belief": dict(self.belief),
                           "away": dict(self.away)}
            return SUCK

        other = "B" if cell == "A" else "A"
        p = self._p_dirty(other)
        gain = p * max(self.horizon - 2, 0)
        self.detail = {"dirty_here": False, "other": other, "p": p, "gain": gain,
                       "k": self.away[other], "rate": self.rate, "span": self.horizon - 2,
                       "cost": self.move_cost, "belief": dict(self.belief),
                       "away": dict(self.away),
                       "action": move_from(cell) if gain > self.move_cost else NOOP}
        self.note = "p(dirty|%s)=%.2f after %d steps away, gain %.1f vs cost %d -> %s" % (
            other, p, self.away[other], gain, self.move_cost,
            "patrol" if gain > self.move_cost else "stay")
        return move_from(cell) if gain > self.move_cost else NOOP

    def state(self):
        return self.note


class Learning:
    """Learning agent: learns a model of the environment.

    Maps onto the four components of the textbook diagram --
    performance element, critic, learning element, problem generator.
    The learning element estimates a regrowth rate per cell, which is what
    lets it patrol B more often than A without being told to.
    """

    name = "Learning (model-based)"

    def __init__(self, move_cost=1, horizon=10, seed=1, **_):
        self.move_cost = move_cost
        self.horizon = horizon
        self.rng = Rng(seed)
        self.belief = {"A": UNKNOWN, "B": UNKNOWN}
        self.away = {"A": 0, "B": 0}
        self.found_dirty = {"A": 0, "B": 0}
        self.steps_away = {"A": 0, "B": 0}
        self.steps = 0
        self.last_score = 0
        self.critic = 0
        self.note = ""
        self.detail = {}

    def rate(self, cell):
        return (self.found_dirty[cell] + 1) / (self.steps_away[cell] + 10)

    def _p_dirty(self, cell):
        if self.belief[cell] == UNKNOWN:
            return 1.0
        return 1.0 - (1.0 - self.rate(cell)) ** self.away[cell]

    def reward(self, score):
        self.critic = score - self.last_score
        self.last_score = score

    def __call__(self, percept):
        cell, status = percept
        self.steps += 1
        for c in CELLS:
            self.away[c] += 1

        if self.belief[cell] != UNKNOWN:
            self.steps_away[cell] += self.away[cell]
            if status == DIRTY:
                self.found_dirty[cell] += 1
        self.away[cell] = 0
        self.belief[cell] = status

        base = {"rate_a": self.rate("A"), "rate_b": self.rate("B"),
                "critic": self.critic, "steps": self.steps}
        if status == DIRTY:
            self.belief[cell] = CLEAN
            self.note = "suck"
            self.detail = dict(base, mode="suck", action=SUCK)
            return SUCK

        other = "B" if cell == "A" else "A"

        eps = 0.35 / (1.0 + 0.25 * self.steps)
        if self.rng.random() < eps:
            self.note = "explore %s (problem generator, eps=%.2f)" % (other, eps)
            self.detail = dict(base, mode="explore", eps=eps, other=other,
                               action=move_from(cell))
            return move_from(cell)

        p = self._p_dirty(other)
        gain = p * max(self.horizon - 2, 0)
        self.detail = dict(base, mode="decide", other=other, p=p, gain=gain,
                           cost=self.move_cost,
                           action=move_from(cell) if gain > self.move_cost else NOOP)
        self.note = "r(%s)~%.3f, p(dirty)=%.2f, gain %.1f vs %d" % (
            other, self.rate(other), p, gain, self.move_cost)
        return move_from(cell) if gain > self.move_cost else NOOP

    def state(self):
        return "learned r(A)~%.3f, r(B)~%.3f (true %.2f / %.2f) | %s" % (
            self.rate("A"), self.rate("B"), REGROW["A"], REGROW["B"], self.note)


class QLearning:
    """Reinforcement learning: learns action values, not a world model.

    The agent only perceives its own cell, so there are four states. That is
    a partially observable environment, and the learned policy cannot be
    optimal because of it.
    """

    name = "Reinforcement learning"
    ACTIONS = (SUCK, "Move", NOOP)

    def __init__(self, move_cost=1, alpha=0.30, gamma=0.90, seed=1, **_):
        self.move_cost = move_cost
        self.alpha = alpha
        self.gamma = gamma
        self.rng = Rng(seed)
        self.Q = {(c, s): [0.0, 0.0, 0.0] for c in CELLS for s in (CLEAN, DIRTY)}
        self.prev = None
        self.last_score = 0
        self.pending = 0.0
        self.steps = 0
        self.detail = {}

    def eps(self):
        return max(0.05, 0.60 / (1.0 + 0.02 * self.steps))

    def reward(self, score):
        self.pending = score - self.last_score
        self.last_score = score

    def __call__(self, percept):
        s = percept
        upd = None
        if self.prev is not None:
            s0, a0 = self.prev
            old = self.Q[s0][a0]
            self.Q[s0][a0] = old + self.alpha * (
                self.pending + self.gamma * max(self.Q[s]) - old)
            upd = (s0, a0, old, self.Q[s0][a0], self.pending)

        eps = self.eps()
        explore = self.rng.random() < eps
        if explore:
            a = self.rng.randrange(3)
        else:
            best = max(self.Q[s])
            a = self.rng.choice([i for i, v in enumerate(self.Q[s]) if v == best])

        self.prev = (s, a)
        self.steps += 1
        self.detail = {"update": upd, "eps": eps, "explore": explore,
                       "row": list(self.Q[s]), "state": s, "choice": a}
        action = self.ACTIONS[a]
        return move_from(s[0]) if action == "Move" else action

    def greedy(self, s):
        # ties within 3% count as undecided; prefer the cheaper action
        best = max(self.Q[s])
        margin = abs(best) * 0.03 + 1e-9
        near = [i for i, v in enumerate(self.Q[s]) if best - v <= margin]
        for i in (2, 0, 1):
            if i in near:
                return i
        return near[0]

    def policy(self):
        return {"[%s, %s]" % s: self.ACTIONS[self.greedy(s)] for s in sorted(self.Q)}

    def state(self):
        return "eps=%.2f" % self.eps()


AGENTS = [TableDriven, SimpleReflex, ModelBasedReflex, GoalBased,
          UtilityBased, Learning, QLearning]


def run(agent_cls, steps=20, dynamic=False, seed=0, move_cost=1, **kwargs):
    """Return the world plus a step-by-step trace."""
    world = World(dynamic=dynamic, seed=seed, move_cost=move_cost)
    agent = agent_cls(move_cost=move_cost, seed=seed + 100, **kwargs)
    trace = []
    for i in range(1, steps + 1):
        percept = world.percept()
        action = agent(percept)
        world.step(action)
        if hasattr(agent, "reward"):
            agent.reward(world.score)
        trace.append({
            "step": i,
            "percept": "[%s, %s]" % percept,
            "action": action,
            "score": world.score,
            "internal": agent.state(),
        })
    return world, agent, trace


def compare(steps=20, dynamic=False, move_cost=1, seeds=24):
    rows = []
    for cls in AGENTS:
        runs = [run(cls, steps=steps, dynamic=dynamic, seed=s, move_cost=move_cost)[0]
                for s in range(seeds)]
        rows.append({
            "agent": cls.name,
            "score": sum(w.score for w in runs) / len(runs),
            "moves": sum(w.moves for w in runs) / len(runs),
        })
    return rows


if __name__ == "__main__":
    for row in compare(steps=120, dynamic=True, move_cost=1):
        print("%-24s %7.1f %6.1f" % (row["agent"], row["score"], row["moves"]))
