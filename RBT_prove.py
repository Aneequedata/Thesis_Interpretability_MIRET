import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from math import ceil
from math import floor
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import time
from scipy.spatial import distance_matrix
import pygraphviz as pgv
import os
import matplotlib.pyplot as plt

# import tikzplotlib
# import tikzplotlib as plt_tikz

if not 'C:\\Program Files\\Graphviz\\bin' in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'


def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


class RBTree():
    def __init__(self, name='RBT', version='multivariate', depth=2):

        self.name = name
        self.version = version
        self.depth = depth

        self.T = self.find_T(self.depth)  # total number of nodes
        floorTb = (floor(self.T / 2))
        self.Tb = [i for i in range(floorTb)]  # branch nodes
        self.Tl = [i for i in range(floorTb, self.T)]  # leaf nodes
        self.parents = [None] + [int(i) for i in np.arange(0, self.find_T(depth - 1),
                                                           0.5)]  # list where the i-th elem is the parent of the i-th node #todo None or -1?
        self.child_sx = [2*t+1 for t in self.Tb]
        self.child_dx = [2*t+2 for t in self.Tb]

        self.Tb_first = self.Tb[:self.find_T(self.depth - 2)]  # branch nodes until the last level (escluso)
        self.Tb_last = self.Tb[self.find_T(self.depth - 2):]  # branch nodes of the last level
        self.Tbl = [list(np.arange(2 ** i - 1, 2 ** (i + 1) - 1)) for i in range(self.depth)]
        self.Tb_level = [list(np.arange(2 ** i - 1, 2 ** (i + 1) - 1)) for i in range(self.depth)]

        self.A, self.Al, self.Ar = self.ancestors()
        self.S, self.Sl, self.Sr = self.sub_leaves()
        self.Sn, self.Sn_l, self.Sn_r = self.sub_nodes()
        # print('subleaves',self.S, self.Sl, self.Sr)

    def find_T(self, depth):
        return pow(2, (depth + 1)) - 1

    def ancestors(self):
        Al = [[] for _ in range(self.T)]
        Ar = [[] for _ in range(self.T)]
        for i in range(self.depth):
            for j in range(self.find_T(i), self.find_T(i + 1)):
                if j % 2 == 0:  # se nodo è pari
                    Al[j] = Al[self.parents[j]]  # aggiungi gli ancestor left del suo genitore
                    Ar[j] = Ar[self.parents[j]] + [
                        self.parents[j]]  # aggiungi il suo genitore più gli ancestor right del suo genitore
                if j % 2 != 0:  # se nodo è dispari
                    Al[j] = Al[self.parents[j]] + [
                        self.parents[j]]  # aggiungi il suo genitore più gli ancestor left del suo genitore
                    Ar[j] = Ar[self.parents[j]]  # aggiungi gli ancestor right del suo genitore
        A = [Al[i] + Ar[i] for i in range(self.T)]  # todo FARLI ORDINATI
        for elem in A:
            elem.sort()
        return A, Al, Ar

    def sub_nodes(self):
        Tb2 = [list(np.arange(2 ** i - 1, 2 ** (i + 1) - 1)) for i in range(self.depth)]
        Sn = [[] for _ in self.Tb_first]
        Sn_l = [[] for _ in self.Tb_first]
        Sn_r = [[] for _ in self.Tb_first]
        for t in self.Tb_first:
            if t == 0:
                Sn[t] = Tb2[1:]
            else:
                a = Sn[self.parents[t]][1:]
                Sn[t] = [elem[floor(len(elem) / 2):] if t % 2 == 0 else elem[:ceil(len(elem) / 2)] for elem in a]
            Sn_l[t] = [elem[:ceil(len(elem) / 2)] for elem in Sn[t]]
            Sn_r[t] = [elem[floor(len(elem) / 2):] for elem in Sn[t]]

        Sn = [[j for e in elem for j in e] for elem in Sn]
        Sn_l = [[j for e in elem for j in e] for elem in Sn_l]
        Sn_r = [[j for e in elem for j in e] for elem in Sn_r]

        return Sn, Sn_l, Sn_r

    def sub_leaves(self):  # particolare!!
        Tb2 = [list(np.arange(2 ** i - 1, 2 ** (i + 1) - 1)) for i in range(self.depth)]
        S = [[] for _ in self.Tb]
        Sr = [[] for _ in self.Tb]
        Sl = [[] for _ in self.Tb]
        for level in Tb2:
            for j in level:
                if j == 0:
                    lp = self.Tl
                    S[j] = lp
                    Sl[j] = lp[:ceil(len(lp) / 2)]
                    Sr[j] = lp[floor(len(lp) / 2):]
                else:
                    if j % 2 != 0:
                        lpl = Sl[self.parents[j]]
                        S[j] = S[self.parents[j]][:ceil((len(S[self.parents[j]]) / 2))]
                        Sl[j] = lpl[:ceil(len(lpl) / 2)]
                        Sr[j] = lpl[floor(len(lpl) / 2):]
                    else:
                        lpr = Sr[self.parents[j]]
                        S[j] = S[self.parents[j]][floor(len(S[self.parents[j]]) / 2):]
                        Sl[j] = lpr[:ceil(len(lpr) / 2)]
                        Sr[j] = lpr[floor(len(lpr) / 2):]
        return S, Sl, Sr

    def model(self, x, y, dataset, level=False, F=None, p=None, f=None, alpha=0.1, FSB=False, FSE=False, B=None, U=None,
              Un=None,
              max_clust=None, flag_clust=False, sym_global=False, eps=1e-3, Ma=1,
              Mh=None):  # warm_start = None, time_limit_ws = None, log_flag_ws = False, concatenate_ws = False):

        self.m = gp.Model()  # initialize the model

        self.x = x
        self.y = y
        self.dataset = dataset
        self.classes, self.counts = np.unique(self.y, return_counts=True)
        self.P = len(self.x)
        self.n = np.shape(self.x)[
            1]  # len(self.x[0]) #todo decidere, potrei anche non usarlo e fare for j in self.features, basta che sia universale per tutto il codice

        self.level = level

        self.F = F
        self.p = p
        if self.p is None:
            self.p = [1 for _ in range(self.P)]
        self.f = f
        self.alpha = alpha

        self.FSB = FSB
        self.FSE = FSE
        self.B = B
        self.U = U
        self.Un = Un
        self.max_clust = max_clust
        self.flag_clust = flag_clust
        self.sym_global = sym_global

        self.eps = eps
        self.Ma = Ma
        self.Mh = self.n + 1

        vars = self.variables()
        self.objective_func(vars)
        self.constraints(vars)

        self.time_ws, self.acc_train_ws, self.acc_test_ws, self.mip_gap_ws = 0, None, None, None

        # self.m.update()
        a, b, z, s, c, d, q_l, q_r,g = vars

        # for i in range(self.P):
        #     q_l[0,i].setAttr("BranchPriority", 100)
        #     q_r[0,i].setAttr("BranchPriority", 100)
        # for t in self.Tb[1:]:
        #     for i in range(self.P):
        #         q_l[t, i].setAttr("BranchPriority", 50)
        #         q_r[t, i].setAttr("BranchPriority", 50)
        #
        # self.m.update()

    def variables(self):

        a = self.m.addVars(self.Tb, self.n, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name='a')
        # a = self.m.addVars(self.Tb, self.n_features, vtype=GRB.INTEGER, lb=-1, ub=1, name='a')

        b = self.m.addVars(self.Tb, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name='b')
        # b = self.m.addVars(self.Tb, vtype=GRB.INTEGER, lb=-1, ub=1, name='b')

        z = self.m.addVars(self.P, self.Tl, vtype=GRB.BINARY,
                           name='z')  # addMVar((30,6)...) ? #point i assigned to leaf node t

        s, c, d = None, None, None

        if self.version == 'c':
            c = self.m.addVars(self.P, vtype=GRB.CONTINUOUS, name='c')  # todo lb=0?

        if self.FSB or self.FSE or self.f is not None:
            s = self.m.addVars(self.Tb, self.n, vtype=GRB.BINARY, name='s')
            if self.FSE:
                d = self.m.addVars(self.depth, self.n, vtype=GRB.BINARY, name='d')

        q_l = self.m.addVars(self.Tb, self.P, vtype=GRB.BINARY, name='q_l')
        q_r = self.m.addVars(self.Tb, self.P, vtype=GRB.BINARY, name='q_r')
        g = self.m.addVars(self.Tb, self.P, vtype=GRB.CONTINUOUS, name='g')
       # g = None # if self.sym_global:
        #     j_min = self.m.addVars(self.Tb, vtype=GRB.CONTINUOUS, name='j_min')
        #     v = self.m.addVars(self.Tb, vtype=GRB.CONTINUOUS, name='v')

        return a, b, z, s, c, d, q_l, q_r, g

    def objective_func(self, vars):

        a, b, z, s, c, d, q_l, q_r, g = vars

        u = [1 if t % 2 == 0 else -1 for t in self.Tl]

        if self.version == 'c':
            self.m.setObjective(quicksum(c[i] for i in range(self.P)), sense=GRB.MINIMIZE)
        else:
            if self.f is not None:
                if not self.level:
                    self.m.setObjective(0.5 * quicksum(
                        self.p[i] * self.y[i] * (self.y[i] - quicksum(u[l - self.Tl[0]] * z[i, l] for l in self.Tl)) for
                        i in
                        range(self.P)) + self.alpha * quicksum(
                        (1 / self.f[t][j]) * s[t, j] for t in self.Tb for j in self.F[t]),
                                        sense=GRB.MINIMIZE)
                else:
                    self.m.setObjective(0.5 * quicksum(
                        self.p[i] * self.y[i] * (self.y[i] - quicksum(u[l - self.Tl[0]] * z[i, l] for l in self.Tl)) for
                        i in range(self.P))
                        + self.alpha * quicksum((1 / self.f[level][j]) * s[t, j] for level in range(self.depth) for t in self.Tb_level[level] for j in self.F[level])
                                        , sense=GRB.MINIMIZE)
            else:
                self.m.setObjective(0.5 * quicksum(
                    self.p[i] * self.y[i] * (self.y[i] - quicksum(u[l - self.Tl[0]] * z[i, l] for l in self.Tl)) for i
                    in
                    range(self.P)), sense=GRB.MINIMIZE)
    #+ 5000*quicksum(g[t,i] for t in self.Tb for i in range(self.P)))

    def constraints(self, vars):  # Constraints

        a, b, z, s, c, d, q_l, q_r, g = vars

        u = [1 if t % 2 == 0 else -1 for t in self.Tl]

        if self.version == 'c':
            self.m.addConstrs(
                c[i] >= self.y[i] * (self.y[i] - quicksum(u[l - self.Tl[0]] * z[i, l] for l in self.Tl)) / 2 for i in
                range(self.P))

        # self.m.addConstrs((quicksum(z[i, t] for t in self.Tl) == 1 for i in range(self.P)),
        #                   name='final_assignment')  # ogni sample va a finire solo in una foglia

        self.m.addConstrs(q_l[t,i] == quicksum(z[i, l] for l in self.Sl[t]) for t in self.Tb for i in range(self.P))
        self.m.addConstrs(q_r[t,i] == quicksum(z[i, l] for l in self.Sr[t]) for t in self.Tb for i in range(self.P))

        self.m.addConstrs(( q_l[0, i]+q_r[0, i] == 1 for i in range(self.P) ), name='final_assignment')  # ogni sample va a finire solo in una foglia

        self.m.addConstrs((quicksum(a[t, j] * self.x[i, j] for j in range(self.n)) + b[t] - self.eps >= - self.Mh * (
                1 - q_r[t,i]) for i in range(self.P) for t in self.Tb), name='routing_right')

        self.m.addConstrs((quicksum(a[t, j] * self.x[i, j] for j in range(self.n)) + b[t] <= self.Mh * (
                1 - q_l[t,i]) for i in range(self.P) for t in self.Tb), name='routing_left')

        self.m.addConstrs(q_l[t,i] == q_l[self.child_sx[t],i] + q_r[self.child_sx[t],i] for t in self.Tb_first for i in range(self.P))
        self.m.addConstrs(q_r[t,i] == q_l[self.child_dx[t],i] + q_r[self.child_dx[t],i] for t in self.Tb_first for i in range(self.P))

       # self.m.addConstrs(g[t, i] <= q_l[t, i] for t in self.Tb for i in range(self.P))
       # self.m.addConstrs(g[t, i] <= q_r[t, i] for t in self.Tb for i in range(self.P))
       # self.m.addConstrs(g[t, i] >= q_l[t, i] + q_r[t, i] - 1 for t in self.Tb for i in range(self.P))
       # self.m.addConstrs(g[t, i] == 0 for t in self.Tb for i in range(self.P))

        if self.FSB or self.FSE or self.f is not None:
            self.m.addConstrs((a[t, j] >= -self.Ma * s[t, j] for t in self.Tb for j in range(self.n)), name='a_ms')
            self.m.addConstrs((a[t, j] <= self.Ma * s[t, j] for t in self.Tb for j in range(self.n)), name='a_ps')
            self.m.addConstr(quicksum(s[t, j] for t in self.Tb for j in range(self.n)) >= 1)

            if self.FSB:
                if self.level:
                    self.m.addConstrs((
                        quicksum(s[t, j] for j in self.F[level]) <= self.B for level in range(self.depth) for t in
                        self.Tb_level[level]), name='budget')
                else:
                    self.m.addConstrs((
                        quicksum(s[t, j] for j in self.F[t]) <= self.B for t in self.Tb), name='budget_node')

        if self.level:
            self.m.addConstrs((
                a[t, j] == 0 for level in range(self.depth) for t in self.Tb_level[level] for j in range(self.n) if
                j not in self.F[level]), name='fs_a')
            self.m.addConstrs((
                s[t, j] == 0 for level in range(self.depth) for t in self.Tb_level[level] for j in range(self.n) if
                j not in self.F[level]), name='fs_s')
        else:
            self.m.addConstrs((a[t, j] == 0 for t in self.Tb for j in range(self.n) if j not in self.F[t]),
                              name='fs_a_node')
            self.m.addConstrs((s[t, j] == 0 for t in self.Tb for j in range(self.n) if j not in self.F[t]),
                              name='fs_s_node')

        if self.sym_global:  # il primo coeff usato valore positivo

            # j == min(j: s[t,j] == 1)
            # self.m.addConstrs(js[t] == [j for j in range(n) if s[t,j] > 0] for t in self.Tb)
            # self.m.addConstrs(j_min[t] == gp.min_(js[t]) for t in self.Tb)
            #
            #  self.m.addConstrs(v[t] == [(j+1)*s[t,j] for j in range(self.n)] for t in self.Tb)
            #  self.m.addConstrs(j_min[t] == gp.min_(v[t]) for t in self.Tb)
            #  self.m.addConstrs(j_min[t] >= self.eps for t in self.Tb)
            #
            #  self.m.addConstrs(a[t, j_min-1] > 0 for t in self.Tb)
            self.m.addConstrs(b[t] >= 0 for t in self.Tb_first)

            # self.m.addConstrs(a[t, j_max_t[t]] > 0 for t in self.Tb)

        if self.U is not None:
            self.m.addConstrs((z[i, l] == z[k, l] for (i, k) in self.U for l in self.Tl), name='proximity')
            if self.max_clust is not None:
                if self.flag_clust:
                    label_clust = self.y[self.max_clust[0]]
                    l_clust_idx = [1 if label_clust > 0 else 0][0]
                    l_clust = self.Tl[l_clust_idx]
                    self.m.addConstrs(z[i, l_clust] == 1 for i in self.max_clust)
                else:
                    self.m.addConstrs(quicksum(z[i, l] for l in self.Tl[:2]) == 1 for i in self.max_clust)

        if self.Un is not None:
            self.m.addConstrs(z[i, l] <= 1 - z[k, l] for (i, k) in self.Un for l in self.Tl)
        # zil 1 -> zik = 0
        # zil = -> zik o 0 o 1

    def solve(self, log_flag=True, time_limit=None, mip_gap=None, cuts=True, presolve=True, feat=None, n_tot=None):

        self.m.Params.LogToConsole = log_flag
        if time_limit is not None:
            self.m.Params.TimeLimit = time_limit
        if mip_gap is not None:
            self.m.Params.MIPGap = mip_gap
        if not cuts:
            self.m.Params.Cuts = 0
        if not presolve:
            self.m.Params.Presolve = 0

       # self.m.Params.NodeLimit = 0
        #self.m.Params.Cuts = -1

        start = time.time()
        self.m.optimize()
        end = time.time()
        print('Time %s:' % self.dataset, end - start)

        if self.m.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            print('Model cannot be solved because it is infeasbile or unbounded')
            print('Optimization was stopped with status %d' % self.m.status)
            # do IIS
            self.m.computeIIS()
            for c in self.m.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
        elif self.m.Status != GRB.OPTIMAL:
            print('Optimization was stopped with status' + str(self.m.Status))
        try:
            obj = self.m.getObjective()
            obj_value = obj.getValue()

            if feat is None:
                self.a = [[self.m.getVarByName('a[%d,%d]' % (t, j)).x for j in range(self.n)] for t in self.Tb]
                self.b = [self.m.getVarByName('b[%d]' % t).x for t in self.Tb]
                self.z = [[self.m.getVarByName('z[%d,%d]' % (i, t)).x for t in self.Tl] for i in range(self.P)]
                # self.s = None
                self.c = None
                self.s = [[self.m.getVarByName('s[%d,%d]' % (t, j)).x for j in range(self.n)] for t in self.Tb]
              #  self.g = [[self.m.getVarByName('g[%d,%d]' % (t, i)).x for i in range(self.P)] for t in self.Tb]
                self.q_l = [[self.m.getVarByName('q_l[%d,%d]' % (t, i)).x for i in range(self.P)] for t in self.Tb]
                self.q_r = [[self.m.getVarByName('q_r[%d,%d]' % (t, i)).x for i in range(self.P)] for t in self.Tb]

                if self.version == 'c':
                    self.c = [self.m.getVarByName('c[%d]' % i).x for i in range(self.P)]

                var_dict = {'q_l': self.q_l, 'q_r': self.q_r, 'a': self.a, 'b': self.b, 'z': self.z, 's': self.s, 'c': self.c}

                F = list(np.unique([j for a_t in self.a for j in range(len(a_t)) if np.abs(a_t[j]) >= 5e-04]))
                NF = len(F)
                Ft = [[j for j in range(len(self.a[t])) if np.abs(self.a[t][j]) >= 5e-04] for t in self.Tb]
                NFt = [len(elem) for elem in Ft]
                n_nodes = self.m.NodeCount
            else:
                print('n0', self.n)
                dict_f = {feat[i]: i for i in range(self.n)}  # o len(feat)
                print('dict_f', dict_f)
                self.n = n_tot
                print('n1', self.n)

                self.a = [
                    [self.m.getVarByName('a[%d,%d]' % (t, dict_f[j])).x if j in feat else 0.0 for j in range(self.n)]
                    for t in self.Tb]
                self.b = [self.m.getVarByName('b[%d]' % t).x for t in self.Tb]
                self.z = [[self.m.getVarByName('z[%d,%d]' % (i, t)).x for t in self.Tl] for i in range(self.P)]
                self.s = None
                self.c = None
                if self.FSB or self.FSE:
                    self.s = [
                        [self.m.getVarByName('s[%d,%d]' % (t, dict_f[j])).x if j in feat else 0 for j in range(self.n)]
                        for t in self.Tb]
                if self.version == 'c':
                    self.c = [self.m.getVarByName('c[%d]' % i).x for i in range(self.P)]

                var_dict = {'a': self.a, 'b': self.b, 'z': self.z, 's': self.s, 'c': self.c}

                F = list(np.unique([j for a_t in self.a for j in range(len(a_t)) if np.abs(a_t[j]) >= 5e-04]))
                NF = len(F)
                Ft = [[j for j in range(len(self.a[t])) if np.abs(self.a[t][j]) >= 5e-04] for t in self.Tb]
                NFt = [len(elem) for elem in Ft]
                n_nodes = self.m.NodeCount

            return n_nodes, obj_value, end - start, self.time_ws, self.m.MIPGap, F, NF, Ft, NFt, var_dict, self.acc_train_ws, self.acc_test_ws, self.mip_gap_ws
        except:
            print('Ottimizzazione non andata a buon fine')
            return None, None, None, None, None, None, None, None, None, None, None, None, None  # todo?

    def proximity(self, x, input=None): #da modificare

        if input is not None:
            self.a, self.b = input

        paths = np.zeros((len(x), self.depth + 1), dtype=object)  # path (sequences of nodes) for each sample i
        for i in range(len(x)):
            if i % 10 == 0:
                print(f"Processing index: {i}, ")
            for d in range(self.depth):
                t = paths[i][d]
                if (np.dot(self.a[t], x[i]) + self.b[t]) <= 1e-6:  # > -1e-12:
                    paths[i][d + 1] = 2 * t + 1
                else:
                    paths[i][d + 1] = 2 * t + 2


        leaves = np.array([elem[-1] for elem in paths])
        m = np.array([[1.0 if leaves[i] == leaves[k] else 0.0 for k in range(len(x))] for i in range(len(x))])
        U = [(i, k) for i in range(len(x)) for k in range(len(x)) if k > i and m[i, k] == 1.0]
        Un = [(i, k) for i in range(len(x)) for k in range(len(x)) if k > i and m[i, k] == 0.0]

        # leaves_vere = [z[i].index(1)+self.Tl[0] for i in range(len(x))]

        return m, U, Un

    def prediction(self, x, input=None):

        # if input is None:
        #     a = [[self.m.getVarByName('a[%d,%d]' % (t, j)).x for j in range(self.n_features)] for t in self.Tb]
        #     b = [self.m.getVarByName('b[%d]' % t).x for t in self.Tb]
        if input is not None:
            self.a, self.b = input

        paths = np.zeros((len(x), self.depth + 1), dtype=object)  # path (sequences of nodes) for each sample i
        for i in range(len(x)):
            for d in range(self.depth):
                t = paths[i][d]
                if (np.dot(self.a[t], x[i]) + self.b[t]) <= 1e-6:  # -1e-12:
                    paths[i][d + 1] = 2 * t + 1
                else:
                    paths[i][d + 1] = 2 * t + 2

        prediction = [1 if paths[i][-1] % 2 == 0 else -1 for i in range(len(x))]
        return prediction

    def performances(self, y, pred):
        # print('y',y)
        # print('pred',pred)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        precision = precision_score(y, pred)
        recall = recall_score(y, pred)

        TN, TP = sum([1 for i in range(len(pred)) if pred[i] == -1 and pred[i] == y[i]]), sum(
            [1 for i in range(len(pred)) if pred[i] == 1 and pred[i] == y[i]]),
        Pm1, P1 = len(y[y == -1]), len(y[y == 1])

        TNR, TPR = TN / Pm1, TP / P1

        AUC = (TNR + TPR) / 2

        return acc, cm, precision, recall, TNR, TPR, AUC

    def compare_proximity(self, thresh_prox_meas, x, U_tree, Un_tree, m_tree, U_rf, Un_rf, m_rf):
        print("Starting compare_proximity method")
        print(
            f"Data sizes - x: {len(x)}, U_tree: {len(U_tree)}, Un_tree: {len(Un_tree)}, m_tree: {len(m_tree)}, U_rf: {len(U_rf)}, Un_rf: {len(Un_rf)}, m_rf: {len(m_rf)}")

        try:
            # Create a binary matrix based on threshold
            print("Creating m_rf_onehot matrix...")
            m_rf_onehot = np.where(m_rf >= thresh_prox_meas, 1.0, 0)

            print("Calculating mcomp...")
            mcomp = ((np.logical_and(m_rf_onehot == m_tree, m_tree == 1).sum() - len(x)) / 2) / (
                    (m_rf_onehot.sum() - len(x)) / 2)

            print(f"mcomp calculated: {mcomp}")

            # Convert lists to sets for efficient intersection
            U_rf_set = set(U_rf)
            U_tree_set = set(U_tree)
            Un_rf_set = set(Un_rf)
            Un_tree_set = set(Un_tree)

            print("Finding intersections for Ucomp...")
            UvsU = U_rf_set.intersection(U_tree_set)
            if len(U_rf) != 0:
                Ucomp = len(UvsU) / len(U_rf)
            else:
                Ucomp = None
            print(f"Ucomp calculated: {Ucomp}")

            print("Finding intersections for Ucomp_n...")
            UvsU_n = Un_rf_set.intersection(Un_tree_set)
            if len(Un_rf) != 0:
                Ucomp_n = len(UvsU_n) / len(Un_rf)
            else:
                Ucomp_n = None
            print(f"Ucomp_n calculated: {Ucomp_n}")

            print("Finished compare_proximity method")
            return mcomp, Ucomp, Ucomp_n

        except Exception as e:
            print(f"Error in compare_proximity method: {e}")
            return None, None, None

    def draw_graph(self, x, y, phase='train', date=None, path='', input=None):

        # if input is None:
        #     w = [[self.m.getVarByName('a[%d,%d]' % (t, j)).x for j in range(self.n_features)] for t in self.Tb]
        #     b = [self.m.getVarByName('b[%d]' % t).x for t in self.Tb]
        if input is not None:
            self.a, self.b = input

        paths = np.zeros((len(x), self.depth + 1), dtype=object)  # path (sequences of nodes) for each sample i
        # points_ombra = {str(t): [] for t in (self.Tb + self.Tl)}
        for i in range(len(x)):
            for d in range(self.depth):
                t = paths[i][d]
                if (np.dot(self.a[t], x[i]) + self.b[t]) <= 1e-6:  # -1e-12: #< 1e-15
                    paths[i][d + 1] = 2 * t + 1
                else:
                    paths[i][d + 1] = 2 * t + 2

        N_m1 = [0 for _ in self.Tb + self.Tl]
        N_1 = [0 for _ in self.Tb + self.Tl]
        for i in range(len(x)):
            for node in self.Tb + self.Tl:
                if node in paths[i]:
                    if y[i] == -1:
                        N_m1[node] += 1
                    else:
                        N_1[node] += 1

        g = pgv.AGraph(directed=True)  # initialize the graph
        nodes = np.arange(self.T)  # nodes = np.append(self.Tb, self.Tl)

        l_feat_used = [[j for j in range(len(self.a[t])) if self.a[t][j] <= -(5e-04) or self.a[t][j] >= 5e-04] for t in
                       self.Tb]
        l_a_used = [[np.round(self.a[t][j], 1) for j in range(len(self.a[t])) if
                     self.a[t][j] <= -(5e-04) or self.a[t][j] >= 5e-04] for t in self.Tb]

        for n in nodes:  # the graph has a node for each node of the tree
            g.add_node(n, shape='circle', size=24)
            if n != 0:
                parent = self.parents[n]
                g.add_edge(parent, n)  # aggiungo arco tra ogni genitore e figlio

        for t in self.Tb:
            g.get_node(t).attr['label'] = '[' + str(N_m1[t]) + ',' + str(N_1[t]) + ']' + '\\n' 'F:' + str(
                l_feat_used[t]) + '\\n' 'a:' + str(l_a_used[t])

        for le in self.Tl:
            if le % 2 == 0:
                g.get_node(le).attr['label'] = '[' + str(N_m1[le]) + ',' + str(N_1[le]) + ']' + '\\n' + '+1'
                g.get_node(le).attr['color'] = 'green'
            else:
                g.get_node(le).attr['label'] = '[' + str(N_m1[le]) + ',' + str(N_1[le]) + ']' + '\\n' + '-1'
                g.get_node(le).attr['color'] = 'red'

        g.layout(prog='dot')
        g.draw(path + 'graph_rbt_%s_d%s_L%s_a%s_%s_FSB%s_gap%s_%s.png' % (
            self.dataset, self.depth, self.level, self.alpha, phase, self.FSB, str(round(self.m.MIPGap, 3)), date))
        img = plt.imread(path + 'graph_rbt_%s_d%s_L%s_a%s_%s_FSB%s_gap%s_%s.png' % (
            self.dataset, self.depth, self.level, self.alpha, phase, self.FSB, str(round(self.m.MIPGap, 3)), date))
        plt.imshow(img)
        plt.show()
        return g

    def draw_plot(self, x, y, size, margin, type='train', date=None, path='',
                  input=None):  # giusta sicuramente fino a depth 3

        if input is not None:
            w, b = input
        else:
            w, b = self.a, self.b

        fig, ax = plt.subplots()
        # get the separating hyperplane
        x1h = [[] for t in self.Tb]
        x2h = [[] for t in self.Tb]
        for t in self.Tb:
            if np.abs(w[t][1]) < 1e-16 and np.abs(w[t][0]) >= 1e-16:
                # print('olle')
                x2h[t] = np.linspace(-0.2, 10.2)
                x1h[t] = [- b[t] / w[t][0] for _ in x2h[t]]
            elif np.abs(w[t][1]) >= 1e-16:
                if t == 0:
                    x1h[t] = np.linspace(-0.2, 10.2)
                    x2h[t] = -w[t][0] / w[t][1] * x1h[t] - b[t] / w[t][1]
                else:
                    ancs, ints = [], []
                    for anc in self.A[t]:
                        if np.abs(w[anc][1]) >= 1e-16 and np.abs(
                                (w[t][0] / w[t][1]) - (w[anc][0] / w[anc][1])) >= 1e-16:
                            x1_int = ((b[anc] / w[anc][1]) - (b[t] / w[t][1])) / (
                                    (w[t][0] / w[t][1]) - (w[anc][0] / w[anc][1]))
                            x2_int = -w[anc][0] / w[anc][1] * x1_int - b[anc] / w[anc][1]  # or t
                            if x1h[anc][0] <= x1_int <= x1h[anc][-1] and max(x2h[anc].min(), -0.2) <= x2_int <= min(
                                    x2h[anc].max(), 10.2):
                                ancs += [anc]
                                ints += [x1_int]
                    if ancs == []:
                        x1h[t] = np.linspace(-0.2, 10.2)
                        x2h[t] = -w[t][0] / w[t][1] * x1h[t] - b[t] / w[t][1]
                    else:
                        if len(ancs) == 1:
                            p, x1_int = ancs[-1], ints[-1]
                            space_dx, space_sx = np.linspace(x1_int, 10.2), np.linspace(-0.2, x1_int)
                            if t in self.Sn_r[p]:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            x1h[t] = space_dx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            x1h[t] = space_sx
                                        else:
                                            x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta
                                            x1h[t] = space_sx
                            else:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            x1h[t] = space_sx
                                        else:
                                            x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta
                                            x1h[t] = space_sx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            # if np.abs(-w[t][0] / w[t][1]) >= np.abs(-w[p][0] / w[p][1]):  # più ripida
                                            x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            x1h[t] = space_dx
                        elif len(ancs) == 2:
                            p, x1_int, x1_int2 = ancs[-1], ints[-1], ints[-2]
                            space_dx, space_sx = np.linspace(x1_int, 10.2), np.linspace(-0.2, x1_int)
                            space_12, space_21 = np.linspace(x1_int, x1_int2), np.linspace(x1_int2, x1_int)
                            if t in self.Sn_r[p]:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                            else:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            # if np.abs(-w[t][0] / w[t][1]) >= np.abs(-w[p][0] / w[p][1]):  # più ripida
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                print('t', t)
                x2h[t] = -w[t][0] / w[t][1] * x1h[t] - b[t] / w[t][1]
        # create a mesh to plot in
        x1_min, x1_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
        x2_min, x2_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2
        x1g, x2g = np.meshgrid(np.arange(x1_min, x1_max, .2), np.arange(x2_min, x2_max, .2))
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, s=25)
        line = [[] for t in self.Tb]
        for t in self.Tb:
            if np.abs(w[t][1]) >= 1e-16 or np.abs(w[t][0]) >= 1e-16:
                line[t], = ax.plot(x1h[t], x2h[t], label='H{}'.format(get_sub('%s' % t)))
        plt.legend()
        ax.axis([x1_min, x2_max, x2_min, x2_max])

        if input is not None:

            plt.savefig(path + 'plot_input_%s_s%s_m%s_%s_d%d_%s_C%s_C1%s_%s.pdf' % (
                self.dataset, str(size), str(margin), type, self.depth, str(self.C), str(self.C1), date))

            # tikzplotlib.save(path + 'plot_input_%s_s%s_m%s_%s_d%d_%s_C%s_C1%s_%s.tex' % (
            #     self.dataset, str(size), str(margin), type, self.depth, str(self.C), str(self.C1), date),
            #                  encoding="utf-8")

        else:

            plt.savefig(path + 'plot_rbt_%s_FSB%s_FSE%s_s%s_m%s_%s_d%d_gap%s_%s.pdf' % (
                self.dataset, self.FSB, self.FSE, str(size), str(margin), type, self.depth, str(self.m.MIPGap), date))

            # tikzplotlib.save(path + 'plot_rbt_%s_FSB%s_FSE%s__s%s_m%s_%s_d%d_gap%s_%s.tex' % (
            #     self.dataset, self.FSB, self.FSE, str(size), str(margin), type, self.depth, str(self.m.MIPGap), date),
            #                  encoding="utf-8")

        # tikzplotlib.save("mytikz.tex",  encoding="utf-8")
        plt.show()

    #####################################################################################################################################################################################################

    def draw_plot4(self, l, x, y, size, margin, type='train', date=None, path=''):  # giusta sicuramente fino a depth 3
        # TODO unire all'altro?

        w = self.a
        b = self.b

        fig, ax = plt.subplots()
        # get the separating hyperplane
        x1h = [[] for t in self.Tb]
        x2h = [[] for t in self.Tb]
        for t in self.Tb:
            if np.abs(w[t][1]) >= 1e-16:
                if t == 0:
                    x1h[t] = np.linspace(-0.2, 10.2)
                    x2h[t] = -w[t][0] / w[t][1] * x1h[t] - b[t] / w[t][1]
                else:
                    ancs, ints = [], []
                    for anc in self.A[t]:
                        if np.abs(w[anc][1]) >= 1e-16 and np.abs(
                                (w[t][0] / w[t][1]) - (w[anc][0] / w[anc][1])) >= 1e-16:
                            x1_int = ((b[anc] / w[anc][1]) - (b[t] / w[t][1])) / (
                                    (w[t][0] / w[t][1]) - (w[anc][0] / w[anc][1]))
                            x2_int = -w[anc][0] / w[anc][1] * x1_int - b[anc] / w[anc][1]  # or t
                            if x1h[anc][0] <= x1_int <= x1h[anc][-1] and max(x2h[anc].min(), -0.2) <= x2_int <= min(
                                    x2h[anc].max(), 10.2):
                                ancs += [anc]
                                ints += [x1_int]
                    if ancs == []:
                        x1h[t] = np.linspace(-0.2, 10.2)
                        x2h[t] = -w[t][0] / w[t][1] * x1h[t] - b[t] / w[t][1]
                    else:
                        if len(ancs) == 1:
                            p, x1_int = ancs[-1], ints[-1]
                            space_dx, space_sx = np.linspace(x1_int, x1h[p][-1]), np.linspace(x1h[p][0], x1_int)
                            if t in self.Sn_r[p]:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            x1h[t] = space_dx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            x1h[t] = space_sx
                                        else:
                                            x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta
                                            x1h[t] = space_sx
                            else:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            x1h[t] = space_sx
                                        else:
                                            x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta
                                            x1h[t] = space_sx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            # if np.abs(-w[t][0] / w[t][1]) >= np.abs(-w[p][0] / w[p][1]):  # più ripida
                                            x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            x1h[t] = space_dx
                        elif len(ancs) == 2:
                            p, x1_int, x1_int2 = ancs[-1], ints[-1], ints[-2]
                            space_dx, space_sx = np.linspace(x1_int, x1h[p][-1]), np.linspace(x1h[p][0], x1_int)
                            space_12, space_21 = np.linspace(x1_int, x1_int2), np.linspace(x1_int2, x1_int)
                            if t in self.Sn_r[p]:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                            else:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            # if np.abs(-w[t][0] / w[t][1]) >= np.abs(-w[p][0] / w[p][1]):  # più ripida
                                            if x1_int >= x1_int2:
                                                x1h[t] = space_21
                                            else:
                                                x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            if x1_int <= x1_int2:
                                                x1h[t] = space_12
                                            else:
                                                x1h[t] = space_dx
                        elif len(ancs) == 3:
                            p, x1_int, x1_int2, x1_int3 = ancs[-1], ints[-1], ints[-2], ints[-3]
                            x_min = min(x1_int2, x1_int3)
                            x_max = max(x1_int2, x1_int3)
                            space_dx, space_sx = np.linspace(x1_int, x1h[p][-1]), np.linspace(x1h[p][0], x1_int)
                            space_1min, space_1max = np.linspace(x1_int, x_min), np.linspace(x1_int, x_max)
                            space_min1, space_max1 = np.linspace(x_min, x1_int), np.linspace(x_max, x1_int)
                            if t in self.Sn_r[p]:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                                        else:
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                            else:
                                if w[p][1] >= 1e-16:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                                        else:
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                                else:
                                    if -w[p][0] / w[p][1] >= 0:
                                        if -w[t][0] / w[t][1] >= -w[p][0] / w[p][1]:  # più ripida
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx
                                        else:  # più piatta o negativa
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                                    else:  # if -w[p][0] / w[p][1] < 0:
                                        if -w[t][0] / w[t][1] <= -w[p][0] / w[p][1]:  # più ripida
                                            # if np.abs(-w[t][0] / w[t][1]) >= np.abs(-w[p][0] / w[p][1]):  # più ripida
                                            if x1_int >= x_max:
                                                x1h[t] = space_max1
                                            elif x1_int >= x_min:
                                                x1h[t] = space_min1
                                            else:
                                                x1h[t] = space_sx
                                        else:  # più piatta o positiva
                                            if x1_int <= x_min:
                                                x1h[t] = space_1min
                                            elif x1_int <= x_max:
                                                x1h[t] = space_1max
                                            else:
                                                x1h[t] = space_dx

                    x2h[t] = -w[t][0] / w[t][1] * x1h[t] - b[t] / w[t][1]
        # create a mesh to plot in
        x1_min, x1_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
        x2_min, x2_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2
        x1g, x2g = np.meshgrid(np.arange(x1_min, x1_max, .2), np.arange(x2_min, x2_max, .2))
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, s=25)
        line = [[] for t in self.Tb]
        for t in self.Tb:
            if np.abs(w[t][1]) >= 1e-16:
                line[t], = ax.plot(x1h[t], x2h[t], label='H{}'.format(get_sub('%s' % t)))
        plt.legend()
        ax.axis([x1_min, x2_max, x2_min, x2_max])

        plt.savefig(path + 'plot4_margot_%s_s%s_m%s_%s_%s_d%d_%s_C%s_C1%s_gap%s_%s.pdf' % (
            self.dataset, str(size), str(margin), type, self.warm_start, self.depth, l, str(self.C), str(self.C1),
            str(self.m.MIPGap), date))

        plt.show()
