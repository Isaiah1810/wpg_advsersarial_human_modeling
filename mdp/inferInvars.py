"""
Data-driven separator ("invariant") inference for action guards.

For every action `act`, learn a quantifier-free first-order formula phi over
the state columns such that

	every state in which `act` was taken		  satisfies   phi
	every state in which `act` was never taken	falsifies   phi

Method (after Koenig, Padon, Immerman & Aiken, "First-Order Quantified
Separators", PLDI'20).  Each row is a structure whose signature has only
0-ary symbols (the state columns), so the quantifier prefix is empty and the
separation problem reduces to the paper's matrix search: find the *smallest*
CNF over a language of atomic formulas that is true on every positive and
false on every negative.  We do that exactly (within a clause-length bound):

  1. Atom language (theory per column type)
	   bool column b		:  b, ¬b
	   numeric column x	 :  x <= c, x >= c, x = c, x != c
							   (c ranges over values observed in the data)
	   numeric pair x, y	:  x <= y, x >= y, x = y, x != y
	   categorical column s :  s = v, s != v
	 Literals with identical truth vectors on the sample are merged,
	 keeping the simplest.
  2. Enumerate candidate clauses (disjunctions of <= max_clause_len literals)
	 that are true on *every* positive and irredundant (each literal is
	 needed to cover some positive).
  3. A clause "covers" a negative if it is false on it.  Choose a set of
	 clauses covering every negative, minimising (#clauses, #literals)
	 lexicographically: an exact 0/1 ILP (scipy.optimize.milp), with greedy
	 set cover as a fallback.
  4. Completeness: any negative no candidate covers gets a "point-exclusion"
	 clause  OR_v (v != n_v), shrunk to an irredundant subset.  It is always
	 sound because, after step 3 of the spec, no positive equals a negative
	 on all state columns.  So a separator always exists and is always found.

With form="auto" (the default) we also learn a CNF separating the negatives
from the positives and negate it, which gives a DNF for phi; the smaller of
the two is returned.  Use form="cnf" for paper-faithful CNF only.

Every result is checked against the data before it is returned.

Caveat: the result is a separator for the observed sample, the best Occam
guess for the action's guard.  It is not an inductive invariant of the
underlying system, and more traces can refute it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["inferInvariants", "Invariant", "Literal"]


# --------------------------------------------------------------------------
# Formula representation
# --------------------------------------------------------------------------

_NEG_OP = {"<=": ">", ">": "<=", ">=": "<", "<": ">=", "==": "!=", "!=": "==",
		   "is": "not", "not": "is"}
_UNI = {"<=": "≤", ">=": "≥", "<": "<", ">": ">", "==": "=", "!=": "≠"}


@dataclass(frozen=True)
class Literal:
	"""An atomic formula or its negation.

	op is one of <=, <, >=, >, ==, != (rhs is a constant, or a column name
	when rhs_is_col is True), or 'is'/'not' for boolean columns.
	"""
	col: str
	op: str
	rhs: object = None
	rhs_is_col: bool = False
	cost: float = 1.0		  # used only to break ties between equal-size formulas

	def negate(self) -> "Literal":
		return Literal(self.col, _NEG_OP[self.op], self.rhs, self.rhs_is_col, self.cost)

	def evaluate(self, df: pd.DataFrame) -> np.ndarray:
		x = df[self.col].to_numpy()
		if self.op == "is":
			return x.astype(bool)
		if self.op == "not":
			return ~x.astype(bool)
		y = df[self.rhs].to_numpy() if self.rhs_is_col else self.rhs
		return {
			"<=": lambda: x <= y, "<": lambda: x < y,
			">=": lambda: x >= y, ">": lambda: x > y,
			"==": lambda: x == y, "!=": lambda: x != y,
		}[self.op]().astype(bool)

	def to_string(self, ascii: bool = False) -> str:
		if self.op == "is":
			return self.col
		if self.op == "not":
			return f"!{self.col}" if ascii else f"¬{self.col}"
		op = self.op if ascii else _UNI[self.op]
		rhs = self.rhs if self.rhs_is_col else _fmt_const(self.rhs)
		return f"{self.col} {op} {rhs}"

	def __str__(self) -> str:
		return self.to_string()


def _fmt_const(v) -> str:
	if isinstance(v, str):
		return repr(v)
	if isinstance(v, (np.integer,)):
		return str(int(v))
	if isinstance(v, (np.floating, float)):
		f = float(v)
		return str(int(f)) if f.is_integer() else repr(f)
	return str(v)


@dataclass
class Invariant:
	"""A quantifier-free formula in CNF (AND of ORs) or DNF (OR of ANDs).

	str(inv) gives the formula; inv.evaluate(df) gives a boolean Series.
	"""
	form: str								  # "cnf" or "dnf"
	groups: List[List[Literal]] = field(default_factory=list)
	n_positive: int = 0						# distinct positive states used
	n_negative: int = 0						# distinct negative states used (after overlap removal)
	n_removed_overlap: int = 0				 # distinct negative states removed as overlapping

	# constant formulas: CNF with no clauses == True; DNF with no cubes == False
	@property
	def is_true(self) -> bool:
		return self.form == "cnf" and not self.groups

	@property
	def size(self) -> Tuple[int, int]:
		return len(self.groups), sum(len(g) for g in self.groups)

	def evaluate(self, df: pd.DataFrame) -> pd.Series:
		n = len(df)
		if self.form == "cnf":
			out = np.ones(n, dtype=bool)
			for clause in self.groups:
				c = np.zeros(n, dtype=bool)
				for lit in clause:
					c |= lit.evaluate(df)
				out &= c
		else:
			out = np.zeros(n, dtype=bool)
			for cube in self.groups:
				c = np.ones(n, dtype=bool)
				for lit in cube:
					c &= lit.evaluate(df)
				out |= c
		return pd.Series(out, index=df.index)

	def to_string(self, ascii: bool = False) -> str:
		AND, OR = (" & ", " | ") if ascii else (" ∧ ", " ∨ ")
		if not self.groups:
			return ("True" if self.form == "cnf" else "False") if ascii else \
				   ("⊤" if self.form == "cnf" else "⊥")
		inner, outer = (OR, AND) if self.form == "cnf" else (AND, OR)
		parts = []
		for g in self.groups:
			s = inner.join(l.to_string(ascii) for l in g)
			parts.append(f"({s})" if len(g) > 1 and len(self.groups) > 1 else s)
		return outer.join(parts)

	def to_z3(self, variables: Optional[dict] = None):
		"""Build a z3 expression (needs `pip install z3-solver`).

		variables maps column name to z3 term; if omitted, Bool/Int/Real/String
		constants are created on the fly from the literal types.
		"""
		import z3  # optional dependency
		variables = dict(variables or {})

		def var(name, sample):
			if name not in variables:
				if isinstance(sample, (bool, np.bool_)) or sample is None:
					variables[name] = z3.Bool(name)
				elif isinstance(sample, (int, np.integer)):
					variables[name] = z3.Int(name)
				elif isinstance(sample, (float, np.floating)):
					variables[name] = z3.Real(name)
				else:
					variables[name] = z3.String(name)
			return variables[name]

		def conv(l: Literal):
			if l.op in ("is", "not"):
				v = var(l.col, None)
				return v if l.op == "is" else z3.Not(v)
			if l.rhs_is_col:
				a, b = var(l.col, 0), var(l.rhs, 0)
			else:
				a = var(l.col, l.rhs)
				r = l.rhs
				b = (z3.StringVal(r) if isinstance(r, str)
					 else int(r) if isinstance(r, (int, np.integer))
					 else float(r))
			return {"<=": a <= b, "<": a < b, ">=": a >= b, ">": a > b,
					"==": a == b, "!=": a != b}[l.op]

		if self.form == "cnf":
			return z3.And([z3.Or([conv(l) for l in c]) for c in self.groups]) \
				if self.groups else z3.BoolVal(True)
		return z3.Or([z3.And([conv(l) for l in c]) for c in self.groups]) \
			if self.groups else z3.BoolVal(False)

	def __str__(self) -> str:
		return self.to_string()

	def __repr__(self) -> str:
		return f"Invariant({self.to_string()!r})"


# --------------------------------------------------------------------------
# Atom language
# --------------------------------------------------------------------------

def _is_bool_col(s: pd.Series) -> bool:
	if pd.api.types.is_bool_dtype(s):
		return True
	if s.dtype == object:
		return all(isinstance(v, (bool, np.bool_)) for v in s)
	return False


def _is_num_col(s: pd.Series) -> bool:
	return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _build_literals(states: pd.DataFrame, relational: bool) -> List[Literal]:
	"""Atom language over the given distinct states.

	Costs only break ties between formulas with the same number of clauses
	and literals: thresholds are preferred to equalities, and equalities
	to disequalities and cross-column relations.
	"""
	lits: List[Literal] = []
	num_cols = []
	for c in states.columns:
		s = states[c]
		if _is_bool_col(s):
			lits += [Literal(c, "is", cost=1.0), Literal(c, "not", cost=1.0)]
		elif _is_num_col(s):
			num_cols.append(c)
			for v in np.sort(s.unique()):
				lits += [Literal(c, "<=", v, cost=1.0), Literal(c, ">=", v, cost=1.0),
						 Literal(c, "==", v, cost=1.05), Literal(c, "!=", v, cost=1.1)]
		else:
			for v in pd.unique(s):
				lits += [Literal(c, "==", v, cost=1.0), Literal(c, "!=", v, cost=1.05)]
	if relational:
		for a, b in combinations(num_cols, 2):
			for op in ("<=", ">=", "==", "!="):
				lits.append(Literal(a, op, b, rhs_is_col=True, cost=1.2))
	return lits


def _mask(arr: np.ndarray) -> int:
	"""Pack a boolean vector into a Python int bitset (bit i = arr[i])."""
	if len(arr) == 0:
		return 0
	return int.from_bytes(np.packbits(arr[::-1].astype(np.uint8)).tobytes(), "big") \
		>> ((-len(arr)) % 8)


def _popcount(x: int) -> int:
	return bin(x).count("1")


# --------------------------------------------------------------------------
# Core: minimal CNF separator
# --------------------------------------------------------------------------

def _learn_cnf(P: pd.DataFrame, N: pd.DataFrame, literals: List[Literal],
			   max_clause_len: int, max_candidates: int,
			   time_limit: Optional[float]) -> List[List[Literal]]:
	"""Smallest CNF (fewest clauses, then literals, then tie-break cost) that is
	true on every row of P and false on every row of N.  P and N must be
	disjoint sets of distinct states."""
	nP, nN = len(P), len(N)
	if nN == 0:
		return []									   # ⊤ separates
	allP, allN = (1 << nP) - 1, (1 << nN) - 1

	# Evaluate literals, merge ones with the same truth vector, drop useless ones.
	best: Dict[Tuple[int, int], Literal] = {}
	for lit in literals:
		pm, nm = _mask(lit.evaluate(P)), _mask(lit.evaluate(N))
		if pm == 0 or nm == allN:   # covers no positive / can never falsify a negative
			continue
		key = (pm, nm)
		if key not in best or lit.cost < best[key].cost:
			best[key] = lit
	lits = sorted(((l, pm, nm) for (pm, nm), l in best.items()),
				  key=lambda t: (t[0].cost, -_popcount(t[1])))

	# Enumerate irredundant sound clauses by DFS.
	# Clause state: pos_sat = positives satisfied, neg_fals = negatives on which
	# every literal so far is false (the negatives the clause would cover).
	cands: Dict[int, Tuple[float, Tuple[int, ...]]] = {}   # neg_fals -> (cost, literal idxs)
	budget = [max_candidates]

	def dfs(start, idxs, pos_sat, neg_fals, cost):
		if budget[0] <= 0:
			return
		for i in range(start, len(lits)):
			l, pm, nm = lits[i]
			if not (pm & ~pos_sat):		 # irredundant: must add a new positive
				continue
			nf = neg_fals & ~nm
			if nf == 0:					 # would cover no negative
				continue
			ps, c, ix = pos_sat | pm, cost + l.cost, idxs + (i,)
			if ps == allP:
				budget[0] -= 1
				old = cands.get(nf)
				if old is None or (len(ix), c) < (len(old[1]), old[0]):
					cands[nf] = (c, ix)
			elif len(ix) < max_clause_len:
				dfs(i + 1, ix, ps, nf, c)

	dfs(0, (), 0, allN, 0.0)

	clauses: List[Tuple[int, float, List[Literal]]] = [
		(nf, c, [lits[i][0] for i in ix]) for nf, (c, ix) in cands.items()]

	# Completeness: point-exclusion clauses for negatives nothing else covers.
	covered = 0
	for nf, _, _ in clauses:
		covered |= nf
	cols = list(P.columns)
	for j in range(nN):
		if covered >> j & 1:
			continue
		row = N.iloc[[j]]
		excl = [Literal(c, "!=", row[c].iloc[0], cost=1.1) if not _is_bool_col(N[c])
				else Literal(c, "not" if bool(row[c].iloc[0]) else "is")
				for c in cols]
		# greedily shrink to an irredundant clause that still satisfies all positives
		pms = [_mask(l.evaluate(P)) for l in excl]
		keep, ps = [], 0
		for k in sorted(range(len(excl)), key=lambda k: -_popcount(pms[k])):
			if pms[k] & ~ps:
				keep.append(k)
				ps |= pms[k]
			if ps == allP:
				break
		assert ps == allP, "overlap removal failed: a positive equals a negative"
		cl = [excl[k] for k in keep]
		nf = allN
		for l in cl:
			nf &= ~_mask(l.evaluate(N))
		clauses.append((nf, sum(l.cost for l in cl), cl))
		covered |= nf

	# Dominance pruning: drop a clause whose cover is a subset of a cheaper-or-equal one.
	clauses.sort(key=lambda t: (-_popcount(t[0]), len(t[2]), t[1]))
	if len(clauses) <= 5000:
		kept = []
		for nf, c, cl in clauses:
			if not any((nf & ~k[0]) == 0 and (len(k[2]), k[1]) <= (len(cl), c) for k in kept):
				kept.append((nf, c, cl))
		clauses = kept

	chosen = _set_cover(clauses, nN, time_limit)
	return [clauses[i][2] for i in chosen]


def _set_cover(clauses, nN, time_limit) -> List[int]:
	"""Exact weighted set cover via MILP; greedy fallback."""
	m = len(clauses)
	BIG = 1000.0   # one clause outweighs any number of literals realistically seen
	w = np.array([BIG + len(cl) * 10.0 + c for _, c, cl in clauses])
	try:
		from scipy.optimize import milp, LinearConstraint, Bounds
		from scipy.sparse import lil_matrix
		A = lil_matrix((nN, m), dtype=np.int8)
		for i, (nf, _, _) in enumerate(clauses):
			j, x = 0, nf
			while x:
				if x & 1:
					A[j, i] = 1
				x >>= 1
				j += 1
		opts = {"time_limit": time_limit} if time_limit else {}
		res = milp(c=w, constraints=LinearConstraint(A.tocsr(), lb=1, ub=np.inf),
				   integrality=np.ones(m), bounds=Bounds(0, 1), options=opts)
		if res.x is not None:
			sel = [i for i in range(m) if res.x[i] > 0.5]
			cov = 0
			for i in sel:
				cov |= clauses[i][0]
			if cov == (1 << nN) - 1:
				return sel
	except Exception:
		pass
	# greedy fallback (ln n approximation)
	remaining, sel = (1 << nN) - 1, []
	while remaining:
		i = max(range(m), key=lambda i: _popcount(clauses[i][0] & remaining) / w[i])
		sel.append(i)
		remaining &= ~clauses[i][0]
	return sel


def _cost(groups: List[List[Literal]]) -> Tuple[int, int, float]:
	return len(groups), sum(len(g) for g in groups), sum(l.cost for g in groups for l in g)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def inferInvariants(explt: pd.DataFrame,
					action_col: str = "action",
					state_cols: Optional[Sequence[str]] = None,
					form: str = "auto",
					max_clause_len: int = 3,
					relational: bool = True,
					max_candidates: int = 200_000,
					time_limit: Optional[float] = 30.0) -> Dict[object, Invariant]:
	"""For each action, infer a formula over the state columns that holds on
	every state where the action was taken and fails on every state where it
	was never taken.

	Parameters
	----------
	explt		  : DataFrame with one action per row.
	action_col	 : name of the action column (default "action").
	state_cols	 : state variables to use (default: every column except action_col).
	form		   : "cnf" (as in the PLDI'20 paper), "dnf", or "auto" (smaller of the two).
	max_clause_len : longest clause/cube enumerated.  Longer ones are used only by the
					 point-exclusion fallback, so a separator is always found.
	relational	 : also use x <= y style atoms between numeric columns.
	max_candidates : cap on enumerated candidate clauses per problem.
	time_limit	 : seconds per MILP solve (None = no limit).

	Returns
	-------
	dict mapping each action to an Invariant (str(inv) is the formula;
	inv.evaluate(df) gives a boolean Series; inv.to_z3() needs z3).
	"""
	if action_col not in explt.columns:
		raise KeyError(f"action column {action_col!r} not in DataFrame")
	if form not in ("cnf", "dnf", "auto"):
		raise ValueError("form must be 'cnf', 'dnf' or 'auto'")
	if state_cols is None:
		state_cols = [c for c in explt.columns if c != action_col]
	state_cols = list(state_cols)
	if not state_cols:
		raise ValueError("no state columns")
	if explt[action_col].isna().any():
		raise ValueError(f"column {action_col!r} has missing actions")
	if explt[state_cols].isna().any().any():
		raise ValueError("state columns contain NaN; fill or drop them first "
						 "(NaN breaks both equality-based de-duplication and comparisons)")

	S = explt[state_cols].reset_index(drop=True)
	A = explt[action_col].reset_index(drop=True)
	literals = _build_literals(S.drop_duplicates(), relational)

	result: Dict[object, Invariant] = {}
	for act in pd.unique(A):					   # actions in order of first appearance
		is_pos = (A == act).to_numpy()
		# Distinct states.  Step 3: drop any negative state that also appears as a positive.
		P = S[is_pos].drop_duplicates().reset_index(drop=True)
		N_all = S[~is_pos].drop_duplicates()
		pos_keys = set(map(tuple, P.itertuples(index=False, name=None)))
		overlap = np.array([t in pos_keys for t in N_all.itertuples(index=False, name=None)],
						   dtype=bool)
		N = N_all[~overlap].reset_index(drop=True)

		options = []
		if form in ("cnf", "auto"):
			options.append(("cnf", _learn_cnf(P, N, literals, max_clause_len,
											  max_candidates, time_limit)))
		if form in ("dnf", "auto"):
			if len(N) == 0:
				options.append(("cnf", []))		 # ⊤
			else:
				# phi = ¬psi, where psi is a CNF true on N and false on P.
				psi = _learn_cnf(N, P, literals, max_clause_len, max_candidates, time_limit)
				options.append(("dnf", [[l.negate() for l in cl] for cl in psi]))
		# Clause count and cube count aren't comparable, so rank by total literals first.
		fm, groups = min(options, key=lambda o: (_cost(o[1])[1], _cost(o[1])[0],
												 _cost(o[1])[2], o[0] != "cnf"))

		inv = Invariant(fm, groups, n_positive=len(P), n_negative=len(N),
						n_removed_overlap=int(overlap.sum()))

		# Check the separator against the data (P and N are disjoint by construction).
		if not inv.evaluate(P).all() or inv.evaluate(N).any():
			raise RuntimeError(f"internal error: separator for {act!r} is inconsistent")
		result[act] = inv
	return result
