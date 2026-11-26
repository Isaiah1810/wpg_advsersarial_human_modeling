"""
Empirical transition probabilities for guarded actions, ready for PRISM.

Given the trace table `explt` (one row per step, the action taken in that
state, runs delimited by the index restarting at 0), the per-action guards
from `inferInvariants`, and the bounded state space, build for every action
`a` a PRISM MDP command

	[a] guard_a -> p_1 : u_1 + p_2 : u_2 + ... ;

where each u_k is a *symbolic* update (keep / set / add / flip per variable)
and p_k is its maximum-likelihood probability (relative frequency, exact
fraction) among the observed transitions s --a--> s'.

Pipeline
--------
1. Transitions.  Row i and row i+1 form a transition (s_i, a_i, s_{i+1}) iff
   they belong to the same run, i.e. index[i+1] == index[i] + 1.  A restart
   (index back to 0) or any gap ends the run.  The last action of each run
   has no observed successor and is not used (it is counted as
   `n_terminal`).

2. Update abstraction.  A PRISM command applies the same update expression
   in *every* state satisfying the guard, so a concrete change s -> s' must
   be lifted to an expression over the pre-state.  Each variable gets one
   encoding per action:

	   bool : keep_set  b' unchanged, or b' = <new value>	 (default)
			  frame	 b' unchanged, or b' = !b
			  set	   b' = <new value>		(always written)
	   int  : delta	 x' = x + d   (d = 0 means unchanged)  (default)
			  keep_set  x' unchanged, or x' = c
			  set	   x' = c				  (always written)

   The encodings are chosen jointly, per action, to minimise the BIC of the
   resulting categorical outcome distribution:

	   n * H(outcomes)  +  (K - 1)/2 * log n		(K = #distinct outcomes)

   That is the most compressible, most deterministic description of the data.
   Ties go to the default encodings.  Example: distance 20->19, 19->17,
   17->16 is {-1: 2, -2: 1} under `delta` but three different constants
   under `set`, so `delta` is chosen.

3. Probabilities.  P(u | a) = #(transitions of a with update u) / #(a).
   These are exact `Fraction`s, written to PRISM as `k/n` so that every
   distribution sums to exactly 1.

4. Soundness for PRISM.
   * `x + d` is clamped to [min, max], e.g. max(distance-2, 0).  The guard
	 may admit states never seen in the data, and PRISM rejects an update
	 that leaves a variable's range.
   * The guard of each action is checked against every observed pre-state
	 of that action.  Violations are counted and reported, which happens
	 when the invariants were learned from different data.
   * Variable and action names are checked against PRISM identifier rules
	 and reserved words.

Assumptions, which you should state alongside the model
-------------------------------------------------------
* Markov / stationarity: the effect of an action depends only on the action
  and the pre-state encoding above, not on history or time.
* Within a guard, the *symbolic* effect distribution is state-independent.
  The guard over-approximates the observed pre-states, so unseen states
  inside the guard inherit the same distribution.
* Different actions can be enabled in the same state, so the result is an MDP
  (nondeterministic choice of action).  A DTMC would additionally need
  P(action | state), the policy.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["transitionProbabilities","TransitionModel", "ActionTransitions",
		   "Outcome", "Update", "VarSpec"]


# --------------------------------------------------------------------------
# State space
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VarSpec:
	name: str
	kind: str					   # "bool" | "int"
	lo: Optional[int] = None
	hi: Optional[int] = None

	def prism_decl(self, init=None) -> str:
		t = "bool" if self.kind == "bool" else f"[{self.lo}..{self.hi}]"
		s = f"{self.name} : {t}"
		if init is not None:
			s += f" init {_prism_const(init)}"
		return s + ";"


def _parse_state_space(state_space: Mapping) -> Dict[str, VarSpec]:
	"""Accepts, per variable:
		 bool | "bool"					   -> boolean
		 (lo, hi) | [lo, hi] | range(lo, hi+1) -> bounded int
		 {"type": "int", "min": lo, "max": hi} / {"type": "bool"}
		 VarSpec
	"""
	specs: Dict[str, VarSpec] = {}
	for name, v in state_space.items():
		if isinstance(v, VarSpec):
			specs[name] = v
		elif v is bool or v == "bool":
			specs[name] = VarSpec(name, "bool")
		elif isinstance(v, range):
			specs[name] = VarSpec(name, "int", v.start, v.stop - 1)
		elif isinstance(v, (tuple, list)) and len(v) == 2:
			lo, hi = int(v[0]), int(v[1])
			specs[name] = VarSpec(name, "int", lo, hi)
		elif isinstance(v, Mapping):
			t = v.get("type", "int")
			if t == "bool":
				specs[name] = VarSpec(name, "bool")
			else:
				specs[name] = VarSpec(name, "int", int(v["min"]), int(v["max"]))
		else:
			raise ValueError(f"cannot parse state-space entry {name!r}: {v!r}")
		s = specs[name]
		if s.kind == "int" and s.lo > s.hi:
			raise ValueError(f"{name}: min {s.lo} > max {s.hi}")
	if not specs:
		raise ValueError("empty state space")
	return specs


_PRISM_RESERVED = set("""A bool clock const ctmc C double dtmc E endinit endinvariant
endmodule endobservables endrewards endsystem false formula filter func F global G
init invariant I int label max mdp min module X nondeterministic observable
observables of Pmax Pmin P pomdp popta probabilistic prob pta rate rewards Rmax Rmin
R S stochastic system true U W""".split())
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(name: str, what: str) -> None:
	if not _IDENT.match(str(name)):
		raise ValueError(f"{what} {name!r} is not a valid PRISM identifier")
	if name in _PRISM_RESERVED:
		raise ValueError(f"{what} {name!r} is a PRISM reserved word; rename it")


def _prism_const(v) -> str:
	if isinstance(v, (bool, np.bool_)):
		return "true" if v else "false"
	if isinstance(v, (int, np.integer)):
		return str(int(v))
	if isinstance(v, (float, np.floating)):
		f = float(v)
		return str(int(f)) if f.is_integer() else repr(f)
	raise TypeError(f"PRISM has no constants of type {type(v).__name__}: {v!r}")


# --------------------------------------------------------------------------
# Updates and outcomes
# --------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Update:
	"""One variable's change.  kind: keep | set | add | flip."""
	var: str
	kind: str
	value: object = None

	def is_keep(self) -> bool:
		return self.kind == "keep"

	def to_prism(self, spec: VarSpec, clamp: bool = True) -> str:
		x = self.var
		if self.kind == "set":
			return f"({x}'={_prism_const(self.value)})"
		if self.kind == "flip":
			return f"({x}'=!{x})"
		if self.kind == "add":
			d = int(self.value)
			e = f"{x}+{d}" if d > 0 else f"{x}-{-d}"
			if clamp:
				e = f"min({e},{spec.hi})" if d > 0 else f"max({e},{spec.lo})"
			return f"({x}'={e})"
		raise ValueError(self.kind)

	def __str__(self) -> str:
		if self.kind == "set":
			return f"{self.var}:={_prism_const(self.value)}"
		if self.kind == "flip":
			return f"{self.var}:=!{self.var}"
		if self.kind == "add":
			return f"{self.var}{int(self.value):+d}"
		return f"{self.var} unchanged"


@dataclass
class Outcome:
	updates: Tuple[Update, ...]		  # non-keep updates only, in variable order
	count: int
	prob: Fraction
	rows: List[int] = field(default_factory=list)   # iloc of the pre-state rows in explt

	def to_prism(self, specs: Mapping[str, VarSpec], clamp: bool = True) -> str:
		if not self.updates:
			return "true"
		return "&".join(u.to_prism(specs[u.var], clamp) for u in self.updates)

	def __str__(self) -> str:
		return ", ".join(map(str, self.updates)) or "no change"


@dataclass
class ActionTransitions:
	action: object
	guard: str							# PRISM syntax
	n_transitions: int
	outcomes: List[Outcome]
	encodings: Dict[str, str]			 # variable -> encoding chosen
	n_terminal: int = 0				   # action taken at the last step of a run (no successor)
	n_guard_violations: int = 0		   # observed pre-states that falsify the guard
	score: float = 0.0					# BIC of the outcome distribution
	terminal: bool = False				# action ends the game (sets the terminal flag)

	def to_prism(self, specs: Mapping[str, VarSpec], clamp: bool = True,
				 indent: str = "	") -> str:
		head = f"{indent}[{self.action}] {self.guard} ->"
		if len(self.outcomes) == 1:
			return f"{head} {self.outcomes[0].to_prism(specs, clamp)};"
		sep = f"\n{indent}	+ "
		body = sep.join(f"{o.prob.numerator}/{o.prob.denominator} : {o.to_prism(specs, clamp)}"
						for o in self.outcomes)
		return f"{head}\n{indent}	  {body};"

	def table(self) -> pd.DataFrame:
		return pd.DataFrame({
			"action": [self.action] * len(self.outcomes),
			"update": [str(o) for o in self.outcomes],
			"count": [o.count for o in self.outcomes],
			"prob": [float(o.prob) for o in self.outcomes],
			"prob_exact": [str(o.prob) for o in self.outcomes],
		})


class TransitionModel(dict):
	"""dict: action -> ActionTransitions, plus PRISM export."""

	def __init__(self, actions, specs, initial_states, unobserved, clamp, terminal_var=None):
		super().__init__(actions)
		self.specs: Dict[str, VarSpec] = specs
		self.initial_states: pd.DataFrame = initial_states
		self.unobserved: List[object] = unobserved	  # guarded actions with no transitions
		self.clamp = clamp
		self.terminal_var: Optional[str] = terminal_var # game-over flag, if any action ends the game

	@property
	def terminal_actions(self) -> List[object]:
		return [a for a, t in self.items() if t.terminal]

	def table(self) -> pd.DataFrame:
		if not self:
			return pd.DataFrame(columns=["action", "update", "count", "prob", "prob_exact"])
		return pd.concat([t.table() for t in self.values()], ignore_index=True)

	def to_prism(self, module_name: str = "agent", init: object = "observed",
				 header_comment: bool = True) -> str:
		"""Full PRISM MDP.

		init: "observed" -> the distinct first states of the runs (an init...endinit
			  block if there is more than one),
			  dict var->value -> one initial state,
			  None -> no init (PRISM uses the type's minimum, or false for bools).
		"""
		_check_ident(module_name, "module name")
		lines = []
		if header_comment:
			lines.append("// Generated by transitionProbabilities(): empirical MLE of action effects.")
			lines.append("// Probabilities are relative frequencies over observed transitions.")
		lines += ["mdp", "", f"module {module_name}"]

		init_states = None
		per_var_init: Dict[str, object] = {}
		if isinstance(init, Mapping):
			per_var_init = dict(init)
		elif init == "observed" and len(self.initial_states):
			if len(self.initial_states) == 1:
				per_var_init = self.initial_states.iloc[0].to_dict()
			else:
				init_states = self.initial_states
		elif init not in (None, "observed"):
			raise ValueError("init must be 'observed', a dict, or None")

		for name, spec in self.specs.items():
			lines.append("	" + spec.prism_decl(per_var_init.get(name)))
		lines.append("")
		for act, t in self.items():
			if t.terminal:
				lines.append(f"	// {act}: ends the game ({t.n_transitions} observed, "
							 f"always as the last step of a run)" if t.n_transitions else
							 f"	// {act}: ends the game (declared; not in the data)")
			else:
				lines.append(f"	// {act}: {t.n_transitions} transitions, "
							 f"{len(t.outcomes)} outcome(s); encodings: "
							 + ", ".join(f"{v}={m}" for v, m in t.encodings.items()))
			if t.n_guard_violations:
				lines.append(f"	// WARNING: {t.n_guard_violations} observed pre-state(s) "
							 f"falsify this guard")
			lines.append(t.to_prism(self.specs, self.clamp))
			lines.append("")
		for act in self.unobserved:
			lines.append(f"	// {act}: guard known but no observed transitions; command omitted")
		if self.terminal_var:
			tv = self.terminal_var
			lines += ["	// game over: absorbing end state",
					  f"	[] {tv} -> true;", ""]
		lines.append("endmodule")
		if self.terminal_var:
			lines += ["", f'label "game_over" = {self.terminal_var};']

		if init_states is not None:
			conj = []
			for _, row in init_states.iterrows():
				conj.append("(" + " & ".join(
					(n if row[n] else f"!{n}") if self.specs[n].kind == "bool"
					else f"{n}={_prism_const(row[n])}" for n in self.specs) + ")")
			lines += ["", "init", "	" + "\n  | ".join(conj), "endinit"]
		return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Guards: Invariant objects (or strings) -> PRISM
# --------------------------------------------------------------------------

_OP_PRISM = {"<=": "<=", "<": "<", ">=": ">=", ">": ">", "==": "=", "!=": "!="}


def _literal_to_prism(lit) -> str:
	if lit.op == "is":
		return lit.col
	if lit.op == "not":
		return f"!{lit.col}"
	rhs = lit.rhs if lit.rhs_is_col else _prism_const(lit.rhs)
	return f"{lit.col}{_OP_PRISM[lit.op]}{rhs}"


def _guard_to_prism(inv) -> str:
	if inv is None:
		return "true"
	if isinstance(inv, str):					   # e.g. inv.to_string(ascii=True)
		s = re.sub(r"(?<![<>!=])==", "=", inv)
		s = re.sub(r"\bTrue\b", "true", s)
		return re.sub(r"\bFalse\b", "false", s)
	groups, form = inv.groups, inv.form
	if not groups:
		return "true" if form == "cnf" else "false"
	inner, outer = ("|", "&") if form == "cnf" else ("&", "|")
	parts = []
	for g in groups:
		s = f" {inner} ".join(_literal_to_prism(l) for l in g)
		parts.append(f"({s})" if len(g) > 1 and len(groups) > 1 else s)
	return f" {outer} ".join(parts)


def _guard_holds(inv, df: pd.DataFrame) -> Optional[np.ndarray]:
	"""Boolean vector, or None if the guard can't be evaluated here."""
	if inv is None:
		return np.ones(len(df), dtype=bool)
	if hasattr(inv, "evaluate"):
		return np.asarray(inv.evaluate(df), dtype=bool)
	try:										   # best effort for string guards
		s = re.sub(r"!(?!=)", "~", inv)
		return np.asarray(df.eval(s), dtype=bool)
	except Exception:
		return None


# --------------------------------------------------------------------------
# Encoding selection
# --------------------------------------------------------------------------

_MODES = {"bool": ("keep_set", "frame", "set"),	   # order = tie-break preference
		  "int": ("delta", "keep_set", "set")}


def _encode(spec: VarSpec, mode: str, old, new) -> Update:
	if mode == "set":
		return Update(spec.name, "set", new)
	if old == new:
		return Update(spec.name, "keep")
	if mode == "keep_set":
		return Update(spec.name, "set", new)
	if mode == "frame":
		return Update(spec.name, "flip")
	if mode == "delta":
		return Update(spec.name, "add", int(new) - int(old))
	raise ValueError(mode)


def _bic(labels: Sequence[tuple]) -> Tuple[float, int]:
	n = len(labels)
	c = Counter(labels)
	nll = -sum(k * math.log(k / n) for k in c.values())
	return nll + 0.5 * (len(c) - 1) * math.log(n), len(c)


def _choose_encodings(enc: Dict[str, Dict[str, List[Update]]], var_order: List[str],
					  n: int, forced: Mapping[str, str], max_bruteforce: int = 729):
	"""Pick one mode per variable minimising BIC of the joint outcome labels."""
	choices = {v: ([forced[v]] if v in forced else list(enc[v])) for v in var_order}

	def labels(assign):
		cols = [enc[v][assign[v]] for v in var_order]
		return [tuple(u for u in row if not u.is_keep()) for row in zip(*cols)]

	def key(assign):
		s, k = _bic(labels(assign))
		pref = sum(list(enc[v]).index(assign[v]) for v in var_order)
		return (round(s, 9), k, pref)

	total = math.prod(len(c) for c in choices.values())
	if total <= max_bruteforce:
		best = min((dict(zip(var_order, combo)) for combo in product(*choices.values())),
				   key=key)
	else:										 # coordinate descent
		best = {v: choices[v][0] for v in var_order}
		bk, improved = key(best), True
		while improved:
			improved = False
			for v in var_order:
				for m in choices[v]:
					if m == best[v]:
						continue
					trial = {**best, v: m}
					tk = key(trial)
					if tk < bk:
						best, bk, improved = trial, tk, True
	return best, labels(best), key(best)[0]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def transitionProbabilities(explt: pd.DataFrame,
							invars: Optional[Mapping[object, object]],
							state_space: Mapping[str, object],
							action_col: str = "action",
							encoding: Optional[Mapping[str, str]] = None,
							clamp: bool = True,
							strict_guards: bool = False,
							terminal_actions: object = "auto",
							terminal_var: str = "done",
							run_ends_terminal: bool = False) -> TransitionModel:
	"""Estimate, for every action, the distribution of symbolic next-state updates.

	Parameters
	----------
	explt		: trace table, one row per step; runs are maximal stretches where
				   the index increases by exactly 1 (a reset to 0 starts a new run).
	invars	   : action -> guard (an `Invariant` from inferInvariants, or a guard
				   string).  Actions missing from it get guard `true` (with a warning).
				   Pass None to use `true` everywhere.
	state_space  : var -> bool | (min, max) | {"type":..,"min":..,"max":..}.  Only
				   these columns are state variables, in this order.
	action_col   : name of the action column.
	encoding	 : optional var -> mode, to force an encoding for every action
				   (bool: keep_set|frame|set, int: delta|keep_set|set).
	clamp		: clamp x+d into [min, max] in the PRISM output (recommended).
	strict_guards: raise, instead of warning, if an observed pre-state falsifies
				   its action's guard.
	terminal_actions : actions that end the game.
				   "auto" (default): every action that is never followed by a next
				   state, i.e. only ever taken as the last step of a run (e.g. fire).
				   An iterable: exactly these actions, with probability 1 (they
				   may also be actions that appear only in invars).
				   None: no game-ending actions (such actions are omitted).
				   A game-ending action sets `terminal_var` to true.  Every command
				   is guarded by !terminal_var, and the end state is absorbing.
	terminal_var : name of the game-over flag added to the module (default "done").
	run_ends_terminal : also treat the last step of every run as ending the game, so
				   actions that sometimes end a run get a "game over" outcome with
				   probability #ends/#taken.  Leave False if runs are cut off by a
				   step limit rather than by the game ending.

	Returns
	-------
	TransitionModel (a dict action -> ActionTransitions) with .table() and
	.to_prism(module_name=...).
	"""
	specs = _parse_state_space(state_space)
	var_order = list(specs)
	for v in var_order:
		_check_ident(v, "variable")
		if v not in explt.columns:
			raise KeyError(f"state variable {v!r} not in DataFrame")
	if action_col not in explt.columns:
		raise KeyError(f"action column {action_col!r} not in DataFrame")
	encoding = dict(encoding or {})
	for v, m in encoding.items():
		if v not in specs or m not in _MODES[specs[v].kind]:
			raise ValueError(f"bad forced encoding {v!r}={m!r}")

	# ---- normalise and validate the state columns ----
	S = explt[var_order].reset_index(drop=True).copy()
	if S.isna().any().any():
		raise ValueError("state columns contain NaN")
	for v, spec in specs.items():
		col = S[v]
		if spec.kind == "bool":
			bad = ~col.isin([True, False, 0, 1])
			if bad.any():
				raise ValueError(f"{v}: non-boolean values {col[bad].unique()[:5]}")
			S[v] = col.astype(bool)
		else:
			vals = pd.to_numeric(col)
			if not np.all(np.mod(vals, 1) == 0):
				raise ValueError(f"{v}: non-integer values")
			vals = vals.astype(np.int64)
			out = (vals < spec.lo) | (vals > spec.hi)
			if out.any():
				raise ValueError(f"{v}: values {sorted(vals[out].unique())[:5]} outside "
								 f"[{spec.lo}..{spec.hi}]")
			S[v] = vals
	A = explt[action_col].reset_index(drop=True)
	if A.isna().any():
		raise ValueError(f"column {action_col!r} has missing actions")
	acts = pd.unique(A)
	for a in acts:
		_check_ident(str(a), "action")

	# ---- transitions within runs ----
	idx = np.asarray(explt.index)
	try:
		nxt_same_run = np.zeros(len(idx), dtype=bool)
		nxt_same_run[:-1] = idx[1:].astype(np.int64) == idx[:-1].astype(np.int64) + 1
	except (TypeError, ValueError):
		raise ValueError("explt index must be integer step numbers (0, 1, 2, ... per run)")
	run_start = np.ones(len(idx), dtype=bool)
	run_start[1:] = ~nxt_same_run[:-1]
	initial_states = S[run_start].drop_duplicates().reset_index(drop=True)

	pre_pos = np.flatnonzero(nxt_same_run)		# rows i with a successor i+1
	records = {v: (S[v].to_numpy(), specs[v]) for v in var_order}

	invars = dict(invars or {})
	explicit_terminal = (None if terminal_actions in (None, "auto")
						 else set(terminal_actions))
	if terminal_actions is not None or run_ends_terminal:
		_check_ident(terminal_var, "terminal variable")
		if terminal_var in specs or terminal_var in explt.columns:
			raise ValueError(f"terminal_var {terminal_var!r} clashes with a column; "
							 f"pass another name")
	END = (Update(terminal_var, "set", True),)		  # the "game over" update

	result: Dict[object, ActionTransitions] = {}
	for a in acts:
		rows_all = np.flatnonzero((A == a).to_numpy())
		rows = np.intersect1d(rows_all, pre_pos)
		n_term = len(rows_all) - len(rows)
		inv = invars.get(a)
		if inv is None and invars:
			warnings.warn(f"no invariant for action {a!r}; using guard 'true'")

		# guard check on every observed pre-state of a
		holds = _guard_holds(inv, S.iloc[rows_all].reset_index(drop=True))
		n_viol = 0 if holds is None else int((~holds).sum())
		if holds is None:
			warnings.warn(f"could not evaluate the guard of {a!r} on the data; unchecked")
		elif n_viol:
			msg = (f"guard of {a!r} is false on {n_viol} of {len(rows_all)} observed "
				   f"pre-states; were the invariants learned on other data?")
			if strict_guards:
				raise ValueError(msg)
			warnings.warn(msg)

		# Game-ending action: never followed by a state (auto), or named explicitly.
		is_terminal = (a in explicit_terminal if explicit_terminal is not None
					   else terminal_actions == "auto" and len(rows) == 0)
		if is_terminal:
			result[a] = ActionTransitions(
				a, _guard_to_prism(inv), len(rows_all),
				[Outcome(END, len(rows_all), Fraction(1), [int(r) for r in rows_all])],
				{}, n_terminal=n_term, n_guard_violations=n_viol, terminal=True)
			continue
		if len(rows) == 0:
			warnings.warn(f"action {a!r} has no observed successor; command omitted")
			continue

		# per-variable encodings under every mode
		enc: Dict[str, Dict[str, List[Update]]] = {}
		for v in var_order:
			col, spec = records[v]
			old, new = col[rows].tolist(), col[rows + 1].tolist()
			enc[v] = {m: [_encode(spec, m, o, nw) for o, nw in zip(old, new)]
					  for m in _MODES[spec.kind]}
		chosen, labels, score = _choose_encodings(enc, var_order, len(rows), encoding)
		label_rows = [int(r) for r in rows]
		if run_ends_terminal and n_term:			# run ended right after a -> game over
			term_rows = [int(r) for r in np.setdiff1d(rows_all, rows)]
			labels = list(labels) + [END] * len(term_rows)
			label_rows += term_rows

		counts = Counter(labels)
		n = len(labels)
		outcomes = []
		for lab, k in sorted(counts.items(), key=lambda t: (-t[1], [str(u) for u in t[0]])):
			outcomes.append(Outcome(lab, k, Fraction(k, n),
									[r for r, l in zip(label_rows, labels) if l == lab]))
		assert sum(o.prob for o in outcomes) == 1

		result[a] = ActionTransitions(a, _guard_to_prism(inv), n, outcomes, chosen,
									  n_terminal=n_term, n_guard_violations=n_viol,
									  score=score)

	# Explicitly terminal actions that only appear in invars (never in the data).
	if explicit_terminal is not None:
		for a in explicit_terminal:
			if a not in result:
				if a not in invars:
					raise ValueError(f"terminal action {a!r} is neither in the data nor in invars")
				_check_ident(str(a), "action")
				result[a] = ActionTransitions(a, _guard_to_prism(invars[a]), 0,
											  [Outcome(END, 0, Fraction(1))], {}, terminal=True)

	# If anything can end the game, add the flag: every command is blocked once it is set.
	used_end = any(END in [o.updates for o in t.outcomes] for t in result.values())
	tv = None
	if used_end:
		tv = terminal_var
		specs = {**specs, tv: VarSpec(tv, "bool")}
		initial_states = initial_states.assign(**{tv: False})
		for t in result.values():
			t.guard = f"!{tv}" if t.guard == "true" else f"!{tv} & ({t.guard})"

	unobserved = [a for a in invars if a not in result]
	return TransitionModel(result, specs, initial_states, unobserved, clamp, tv)
