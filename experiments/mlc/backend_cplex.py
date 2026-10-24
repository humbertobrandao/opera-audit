"""Backend CPLEX do protocolo MLC v3 (solver de registro).

CplexLabelProblem herda toda a construcao do modelo de LabelProblem
(audit_core2) e troca apenas a resolucao: CPLEX, 1 thread, TL 600 s,
mip gap 0. Tres modos de run:
  - benchmark: sem callbacks (tempos, nos);
  - instrumentado: MIPInfoCallback grava trajetoria (t, LB_U, UB_U)
    para T_veredito / T_Delta / T_opt;
  - raiz: NodeLimit=0, devolve o dual bound ao fim do no raiz.
"""
from __future__ import annotations
import time
import numpy as np
import cplex
from cplex.callbacks import MIPInfoCallback
from audit_core2 import LabelProblem

TL_SECONDS = 600.0


class _Traj(MIPInfoCallback):
    def __call__(self):
        rec = (time.perf_counter() - self.t0,
               self.get_incumbent_objective_value() if self.has_incumbent() else None,
               self.get_best_objective_value())
        self.rows.append(rec)


class _Res:
    """Imita o resultado do scipy.milp para o _extract herdado."""
    def __init__(self, x, fun, nodes, status):
        self.x = np.array(x); self.fun = fun
        self.mip_node_count = nodes; self.status = 0
        self.cplex_status = status


class CplexLabelProblem(LabelProblem):
    mode = "benchmark"          # benchmark | instrument | root
    last_traj = None
    last_status = None

    def _cplex_solve(self, c, cons, bounds, nv, relax, mode):
        A = cons.A.tocoo() if hasattr(cons.A, "tocoo") else None
        lb, ub = np.asarray(cons.lb, float), np.asarray(cons.ub, float)
        m = self.m
        cpx = cplex.Cplex()
        cpx.set_log_stream(None); cpx.set_results_stream(None)
        cpx.set_warning_stream(None); cpx.set_error_stream(None)
        if relax:
            types = ""
            cpx.variables.add(obj=list(map(float, c)),
                              lb=list(map(float, bounds.lb)),
                              ub=list(map(float, bounds.ub)))
        else:
            tt = ["B"] * m + ["I"] + ["B"] * (nv - m - 1)
            cpx.variables.add(obj=list(map(float, c)),
                              lb=list(map(float, bounds.lb)),
                              ub=list(map(float, bounds.ub)), types=tt)
        rows, senses, rhs = [], [], []
        Acsr = cons.A.tocsr()
        for i in range(Acsr.shape[0]):
            sl = Acsr[i]
            idx = sl.indices.tolist(); val = sl.data.tolist()
            if np.isinf(ub[i]):        # >= lb
                senses.append("G"); rhs.append(float(lb[i]))
            elif np.isinf(lb[i]):      # <= ub
                senses.append("L"); rhs.append(float(ub[i]))
            else:                      # igualdade (lb == ub por construcao)
                senses.append("E"); rhs.append(float(lb[i]))
            rows.append([idx, val])
        cpx.linear_constraints.add(lin_expr=rows, senses="".join(senses),
                                   rhs=rhs)
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.parameters.threads.set(1)
        cpx.parameters.timelimit.set(TL_SECONDS)
        cpx.parameters.mip.tolerances.mipgap.set(0.0)
        if mode == "root":
            cpx.parameters.mip.limits.nodes.set(0)
        # warm start opcional (min|J|: solucao da auditoria principal,
        # viavel por construcao => sempre ha incumbente antes do TL)
        ws = getattr(self, "_mip_start", None)
        if ws is not None and not relax:
            cpx.MIP_starts.add(
                cplex.SparsePair(list(range(nv)), [float(v) for v in ws]),
                cpx.MIP_starts.effort_level.auto)
        traj = None
        if mode == "instrument":
            traj = cpx.register_callback(_Traj)
            traj.rows, traj.t0 = [], time.perf_counter()
        t0 = time.perf_counter()
        cpx.solve()
        dt = time.perf_counter() - t0
        st = cpx.solution.get_status_string()
        stn = cpx.solution.get_status()
        self.last_traj = traj.rows if traj is not None else None
        if mode == "root":
            # nao sobrescrever last_status: o status do no raiz ("node limit
            # exceeded") e informacao do root bound, nao do bloco.
            return {"bound": cpx.solution.MIP.get_best_objective(),
                    "seconds": dt, "status": st}
        # estados: otimo certificado / TL com incumbente / sem solucao viavel
        if not cpx.solution.is_primal_feasible():
            raise RuntimeError(
                f"CPLEX ended with no feasible solution (status: {st}; "
                f"TL={TL_SECONDS:.0f}s) - treat the cell as resource-limited")
        # 1 = CPX_STAT_OPTIMAL (LP); 101/102 = MIP otimo (gap 0)
        self._note_status(stn in (1, 101, 102))
        if relax:
            return _Res(cpx.solution.get_values(),
                        cpx.solution.get_objective_value(), 0, st), dt
        nodes = cpx.solution.progress.get_num_nodes_processed()
        return _Res(cpx.solution.get_values(),
                    cpx.solution.get_objective_value(), nodes, st), dt

    def _run(self, relax=False, **kw):
        c, cons, bounds, nv = self._model(**kw)
        return self._cplex_solve(c, cons, bounds, nv, relax, self.mode)

    def root_bound(self):
        """Dual bound ao fim do no raiz (definicao congelada, RD1/§14)."""
        c, cons, bounds, nv = self._model()
        out = self._cplex_solve(c, cons, bounds, nv, False, "root")
        return {"root_bound": self.U0 - out["bound"],
                "seconds": out["seconds"], "status": out["status"]}

    # ---- trajetoria: solve instrumentado da auditoria principal ----
    def solve_instrumented(self):
        old = self.mode
        self.mode = "instrument"
        try:
            s = self.solve()
        finally:
            self.mode = old
        rows = self.last_traj or []
        # minimizacao interna: U = U0 - obj
        tv = td = None
        for (tt, inc, bnd) in rows:
            LB = (self.U0 - inc) if inc is not None else -np.inf
            UB = self.U0 - bnd
            if tv is None and (LB > 0 or UB <= 0):
                tv = tt
            if td is None and (LB > self.U0 or UB <= self.U0):
                td = tt
        s["T_verdict"] = tv; s["T_delta"] = td
        s["traj_points"] = len(rows)
        s["traj"] = rows[:2000]
        return s
