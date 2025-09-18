from __future__ import annotations

import collections
import dataclasses
import itertools
from typing import Iterator, Any, Iterable, Callable, Literal, cast

import numpy as np
import pygltflib
import stim

import gen


@dataclasses.dataclass(frozen=True)
class Pos:
    xy: complex
    t: float

    def _key(self) -> tuple[float, float, float]:
        return self.xy.real, self.xy.imag, self.t

    def __lt__(self, other):
        if not isinstance(other, Pos):
            return NotImplemented
        return self._key() < other._key()

    def __neg__(self) -> Pos:
        return Pos(-self.xy, -self.t)

    def __pos__(self) -> Pos:
        return self

    def __add__(self, other: Pos) -> Pos:
        if not isinstance(other, Pos):
            return NotImplemented
        return Pos(self.xy + other.xy, self.t + other.t)

    def __mul__(self, other: int) -> Pos:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Pos(self.xy * other, self.t * other)

    __rmul__ = __mul__

    def __sub__(self, other: Pos) -> Pos:
        if not isinstance(other, Pos):
            return NotImplemented
        return Pos(self.xy - other.xy, self.t - other.t)

    def steps(self) -> int:
        return int(abs(self.xy.real)) + int(abs(self.xy.imag)) + int(abs(self.t))

    def __bool__(self) -> bool:
        return bool(self.xy) or bool(self.t)

    def cross(self, other: Pos) -> Pos:
        x1 = self.xy.real
        x2 = other.xy.real
        y1 = self.xy.imag
        y2 = other.xy.imag
        z1 = self.t
        z2 = other.t
        x3 = y1 * z2 - y2 * z1
        y3 = z1 * x2 - z2 * x1
        z3 = x1 * y2 - x2 * y1
        return Pos(x3 + y3 * 1j, z3)

    def triconj(self) -> Pos:
        x = self.xy.real
        y = self.xy.imag
        z = self.t
        x, y, z = z, x, y
        return Pos(x + y * 1j, z)

    def unit(self) -> Pos:
        bx = self.xy.real != 0
        by = self.xy.imag != 0
        bt = self.t != 0
        if bx + by + bt != 1:
            raise ValueError(f"{self!r}.unit() {bx=} {by=} {bt=}")
        if bx:
            return Pos(xy=self.xy.real / abs(self.xy.real), t=0)
        elif by:
            return Pos(xy=1j * self.xy.imag / abs(self.xy.imag), t=0)
        else:
            return Pos(xy=0, t=self.t // abs(self.t))


def name_order_key(s: str) -> Any:
    if s == "X":
        return 0, "X"
    elif s == "Z":
        return 0, "Z"
    elif s.endswith("_in") or s.startswith("in_"):
        return 1, s
    elif not (s.endswith("_out") or s.startswith("out_")):
        return 2, s
    else:
        return 3, s


def gaussian_eliminate_stabilizers(
    stabilizers: list[stim.PauliString], order: Iterable[int]
) -> int:
    solved = 0
    for k in order:
        for basis in [3, 1]:
            for row in range(solved, len(stabilizers)):
                p = stabilizers[row][k]
                if p and p != basis:
                    if row != solved:
                        a, b = stabilizers[row], stabilizers[solved]
                        stabilizers[row], stabilizers[solved] = b, a
                    break
            else:
                continue
            for row in range(len(stabilizers)):
                p = stabilizers[row][k]
                if row != solved and p and p != basis:
                    stabilizers[row] *= stabilizers[solved]
            solved += 1
    return solved


class _PartialTensor:
    def __init__(self, contents: np.ndarray, ports: list[Any]):
        self.contents = contents
        self.ports = ports


@dataclasses.dataclass(frozen=True)
class ZXLatticeGraph:
    nodes: tuple[tuple[str, Pos], ...]
    edges: tuple[tuple[int, int], ...]

    def to_tensor(
        self, *, port_order_key: Callable[[str, Pos], Any] = lambda _, p: p
    ) -> np.ndarray:
        n2edge_indices = collections.defaultdict(list)
        for k, (n1, n2) in enumerate(self.edges):
            n2edge_indices[n1].append(k)
            n2edge_indices[n2].append(k)

        groups = collections.defaultdict(dict)
        externals = []
        port2tensors: dict[Any, list[_PartialTensor]] = collections.defaultdict(list)
        all_tensors = []
        for n, (name, pos) in enumerate(self.nodes):
            deg = len(n2edge_indices[n])
            ports = list(n2edge_indices[n])
            if name == "X":
                v = np.zeros(shape=1 << deg, dtype=np.complex64)
                for k in range(1 << deg):
                    if k.bit_count() & 1 == 0:
                        v[k] = 1
                v.shape = (2,) * deg
            elif name == "Z":
                v = np.zeros(shape=(2,) * deg, dtype=np.complex64)
                v[(0,) * deg] = 1
                v[(1,) * deg] = 1
            elif name == "H":
                if deg != 2:
                    raise NotImplementedError(f"{name=} {deg=}")
                v = np.array([[1, 1], [1, -1]], dtype=np.complex64)
            elif name.startswith("|") and name.endswith(">"):
                if name.startswith("|ccx") and name[-4:] in ["[0]>", "[1]>", "[2]>"]:
                    if deg != 1:
                        raise NotImplementedError(f"{name=} {deg=}")
                    assert int(name[-3]) not in groups[name[:-4] + ">"]
                    groups[name[:-4] + ">"][int(name[-3])] = (name, pos, ports)
                    continue
                elif (
                    name.startswith("|ccz")
                    and name.endswith(">")
                    or name.startswith("ccz")
                    and name.endswith(">")
                ):
                    if deg != 1:
                        raise NotImplementedError(f"{name=} {deg=}")
                    g = groups[name]
                    g[len(g)] = (name, pos, ports)
                    continue
                else:
                    raise NotImplementedError(f"Unrecognized state: {name=}")
            else:
                if deg != 1:
                    raise NotImplementedError(f"{name=} {deg=}")
                v = np.zeros(shape=(2, 2), dtype=np.complex64)
                v[(0, 0)] = 1
                v[(1, 1)] = 1
                externals.append((name, pos))
                ports.append((name, pos))
            p = _PartialTensor(contents=v, ports=ports)
            all_tensors.append(p)
            for port in p.ports:
                port2tensors[port].append(p)

        for gk, gv in groups.items():
            if gk.startswith("|ccx"):
                assert len(gv) == 3
                a = gv[0]
                b = gv[1]
                c = gv[2]
                p = _PartialTensor(
                    contents=np.array([[[1, 0], [1, 0]], [[1, 0], [0, 1]]]),
                    ports=a[2] + b[2] + c[2],
                )
                all_tensors.append(p)
                for port in p.ports:
                    port2tensors[port].append(p)

            elif gk.startswith("|ccz") or gk.startswith("ccz"):
                if len(gv) != 3:
                    raise ValueError(
                        f"The ccz state label '{gk}' occurs {len(gv)} times, instead of exactly 3 times.\n"
                        f"Note you can use variations like |ccz5⟩ and |ccz6⟩."
                    )
                assert len(gv) == 3
                a = gv[0]
                b = gv[1]
                c = gv[2]
                p = _PartialTensor(
                    contents=np.array([[[1, 1], [1, 1]], [[1, 1], [1, -1]]]),
                    ports=a[2] + b[2] + c[2],
                )
                all_tensors.append(p)
                for port in p.ports:
                    port2tensors[port].append(p)

            else:
                raise NotImplementedError(f"{gk=}")

        for k, (n1, n2) in enumerate(self.edges):
            tensor1, tensor2 = port2tensors[k]
            if tensor1 is tensor2:
                n1 = len(tensor1.ports)
                axes1 = [chr(ord("a") + k) for k in range(n1)]
                k1 = tensor1.ports.index(k)
                k2 = tensor1.ports.index(k, k1 + 1)
                axes1[k1] = "Z"
                axes1[k2] = "Z"
                axes1 = "".join(axes1)
                axes3 = axes1.replace("Z", "")
                tensor1.contents = np.einsum(f"{axes1}->{axes3}", tensor1.contents)
                tensor1.ports.pop(k2)
                tensor1.ports.pop(k1)
            else:
                n1 = len(tensor1.ports)
                n2 = len(tensor2.ports)
                axes1 = [chr(ord("a") + k) for k in range(n1)]
                axes2 = [chr(ord("a") + k) for k in range(n1, n1 + n2)]
                k1 = tensor1.ports.index(k)
                k2 = tensor2.ports.index(k)
                axes1[k1] = "Z"
                axes2[k2] = "Z"
                axes1 = "".join(axes1)
                axes2 = "".join(axes2)
                axes3 = (axes1 + axes2).replace("Z", "")
                tensor1.contents = np.einsum(
                    f"{axes1},{axes2}->{axes3}", tensor1.contents, tensor2.contents
                )
                tensor1.ports.pop(k1)
                tensor2.ports.pop(k2)
                tensor1.ports.extend(tensor2.ports)
                for port in tensor2.ports:
                    port2tensors[port].remove(tensor2)
                    port2tensors[port].append(tensor1)
                    tensor2.contents = None
                    tensor2.ports = None

        final_tensors = [t for t in all_tensors if t.contents is not None]
        if len(final_tensors) != 1:
            raise NotImplementedError("Disconnected graph.")
        (final_tensor,) = final_tensors
        assert sorted(final_tensor.ports) == sorted(externals)
        desired_order = sorted(externals, key=lambda e: port_order_key(e[0], e[1]))
        desired_order_indices = [final_tensor.ports.index(k) for k in desired_order]
        v = final_tensor.contents.transpose(desired_order_indices)
        norm = np.linalg.norm(v)
        v /= norm
        return v

    def to_function(self):
        pos2k = {pos: k for k, (_, pos) in enumerate(self.nodes)}
        nk2e = collections.defaultdict(list)
        for k1, k2 in self.edges:
            nk2e[k1].append(k2)
            nk2e[k2].append(k1)

        port_nodes = set()
        for name, pos in self.nodes:
            k1 = pos2k[pos]
            k2s = nk2e[k1]
            if name != "X" and name != "Z":
                port_nodes.add(k1)
                assert len(k2s) == 1, "port must be degree 1"

        e2k = {}
        edge_index_to_port_node_index = {}
        for k1, k2 in self.edges:
            if k1 not in port_nodes and k2 not in port_nodes:
                edge_index = len(e2k)
                e2k[(k1, k2)] = edge_index
        for k1, k2 in self.edges:
            if (k1, k2) not in e2k:
                edge_index = len(e2k)
                e2k[(k1, k2)] = edge_index
                if k1 in port_nodes:
                    edge_index_to_port_node_index[edge_index] = self.nodes[k1][0]
                else:
                    edge_index_to_port_node_index[edge_index] = self.nodes[k2][0]
        for k1, k2 in self.edges:
            e2k[(k2, k1)] = e2k[(k1, k2)]

        stabilizers: list[stim.PauliString] = []
        for name, pos in self.nodes:
            k1 = pos2k[pos]
            k2s = nk2e[k1]
            if name == "X" or name == "Z":
                other = "Z" if name == "X" else "X"
                for k in range(len(k2s) - 1):
                    ps = stim.PauliString(len(self.edges))
                    ps[e2k[(k1, k2s[k])]] = name
                    ps[e2k[(k1, k2s[k + 1])]] = name
                    stabilizers.append(ps)
                ps = stim.PauliString(len(self.edges))
                for k2 in k2s:
                    ps[e2k[(k1, k2)]] = other
                stabilizers.append(ps)

        solved = gaussian_eliminate_stabilizers(
            stabilizers,
            [k for k in range(len(self.edges)) if k not in edge_index_to_port_node_index],
        )

        edge_order = sorted(
            edge_index_to_port_node_index.items(), key=lambda e: name_order_key(e[1])
        )
        ee = []
        for k, v in edge_order:
            ee.append(k)
        outer_stabilizers = []
        for s in stabilizers[solved:]:
            ps = stim.PauliString(len(port_nodes))
            for k in range(len(port_nodes)):
                ps[k] = s[ee[k]]
            outer_stabilizers.append(ps)

        gaussian_eliminate_stabilizers(outer_stabilizers, range(len(outer_stabilizers)))
        num_inputs = 0
        while num_inputs < len(edge_order) and (
            edge_order[num_inputs][1].startswith("in_") or edge_order[num_inputs][1].endswith("_in")
        ):
            num_inputs += 1

        flows = []
        for s in outer_stabilizers:
            flows.append(stim.Flow(input=s[:num_inputs], output=s[num_inputs:]))

        return flows

    def after_shrink(self) -> ZXLatticeGraph:
        x2x = {x: k for k, x in enumerate(sorted({pos.xy.real for _, pos in self.nodes}))}
        y2y = {y: k for k, y in enumerate(sorted({pos.xy.imag for _, pos in self.nodes}))}
        t2t = {t: k for k, t in enumerate(sorted({pos.t for _, pos in self.nodes}))}

        def p2p(pos: Pos) -> Pos:
            return Pos(x2x[pos.xy.real] + 1j * y2y[pos.xy.imag], t2t[pos.t])

        return ZXLatticeGraph(tuple((s, p2p(p)) for s, p in self.nodes), self.edges)

    def __str__(self) -> str:
        min_t = min(pos.t for _, pos in self.nodes)
        max_t = max(pos.t for _, pos in self.nodes)
        depth = max_t - min_t + 1
        diagram = {}
        z_buffer = collections.defaultdict(lambda: min_t)
        longest_name = max(len(name) for name, _ in self.nodes)
        dx = (longest_name + depth + 2) * depth
        dy = dx * 1j
        dt = (longest_name + 1) * (1 + 1j)

        def project(pos: Pos) -> complex:
            return int(pos.xy.real) * dx + int(pos.xy.imag) * dy + pos.t * dt

        def draw(out: complex, t: int, c: str):
            if z_buffer[out] <= t:
                diagram[out] = c
                z_buffer[out] = t

        for k1, k2 in self.edges:
            _, p1 = self.nodes[k1]
            _, p2 = self.nodes[k2]
            dp = p2 - p1
            u = dp.unit()
            unit_char = "-"
            if u.xy.imag:
                unit_char = "|"
            if u.t:
                unit_char = "\\"
            for k in range(dp.steps()):
                ss = p1 + u * k
                v1 = project(ss)
                v2 = project(ss + u)
                dv = v2 - v1
                assert dv.real == 0 or dv.imag == 0 or abs(dv.real) == abs(dv.imag)
                steps = int(max(abs(dv.real), abs(dv.imag)))
                if steps == 0:
                    for j in range(steps + 1):
                        draw(v1, ss.t, unit_char)
                else:
                    du = dv / steps
                    du = round(du.real) + round(du.imag) * 1j
                    assert du * steps == dv
                    for j in range(steps + 1):
                        draw(v1 + du * j, ss.t, unit_char)
        for name, pos in self.nodes:
            pos2 = project(pos)
            for k, c in enumerate(name):
                draw(pos2 + k, pos.t, c)

        min_row = int(min(k.imag for k in diagram.keys()))
        max_row = int(max(k.imag for k in diagram.keys()))
        min_col = int(min(k.real for k in diagram.keys()))
        max_col = int(max(k.real for k in diagram.keys()))
        lines = []
        for row in range(min_row, max_row + 1):
            line = []
            for col in range(min_col, max_col + 1):
                line.append(diagram.get(row * 1j + col, " "))
            lines.append("".join(line).rstrip())
        return "\n".join(lines)

    @staticmethod
    def from_text(graph: str):
        dat = {}
        for row, line in enumerate(graph.splitlines()):
            for col, c in enumerate(line):
                dat[row * 1j + col] = c
        name_chars_2 = set(
            "_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]~!@#$%^&*()<>,."
        )
        literal_dat = {}
        for k, v in dat.items():
            if v == "|" and dat.get(k + 1j, " ") == " " and dat.get(k - 1j, " ") == " ":
                literal_dat[k] = v
                dat[k] = "|literal"
            elif v in name_chars_2:
                literal_dat[k] = v

        def local_neighbors(center: complex) -> Iterator[tuple[complex, complex, Pos]]:
            if dat.get(center) == "-":
                for d in [-1, +1]:
                    if dat.get(center + d) == "|":
                        if dat.get(center + 2 * d) == "-":
                            yield center + 2 * d, d, Pos(xy=2 * d, t=0)
                        else:
                            raise NotImplementedError("Ambiguous - crossing |.")
                    elif dat.get(center + d) == "\\":
                        raise NotImplementedError("- crossing \\")
                    else:
                        yield center + d, d, Pos(xy=d, t=0)
            elif dat.get(center) == "|":
                for d in [-1j, +1j]:
                    if dat.get(center + d) == "-":
                        if dat.get(center + 2 * d) == "|":
                            yield center + 2 * d, d, Pos(xy=d * 2, t=0)
                        else:
                            raise NotImplementedError("Ambiguous crossing.")
                    elif dat.get(center + d) == "\\":
                        raise NotImplementedError("\\ crossing |")
                    else:
                        yield center + d, d, Pos(xy=d, t=0)
            elif dat.get(center) == "\\":
                for dt in [-1, +1]:
                    d = dt * (1j + 1)
                    if dat.get(center + d) == "-":
                        raise NotImplementedError("\\ crossing -")
                    elif dat.get(center + d) == "|":
                        raise NotImplementedError("\\ crossing |")
                    else:
                        yield center + d, d, Pos(xy=0, t=dt)
            elif center in literal_dat:
                for d in [+1, -1]:
                    if dat.get(center + d) == "-" or center + d in literal_dat:
                        yield center + d, d, Pos(xy=d, t=0)
                for d in [+1j, -1j]:
                    if dat.get(center + d) == "|":
                        yield center + d, d, Pos(xy=d, t=0)
                for dt in [-1, +1]:
                    d = dt * (1j + 1)
                    if dat.get(center + d) == "\\":
                        yield center + d, d, Pos(xy=0, t=dt)
            elif dat.get(center) == " ":
                pass
            else:
                raise NotImplementedError(f"{dat.get(center)=}")

        # Find named nodes and map their character positions to the name
        c2rep: dict[complex, complex] = {}
        named: dict[complex, str] = {}
        volume_depth_start_points: dict[complex, int] = {}
        for c, v in dat.items():
            if c in literal_dat and c - 1 not in literal_dat:
                r = c
                while r in literal_dat:
                    r += 1
                locs = [v + c.imag * 1j for v in range(int(c.real), int(r.real))]
                rep = locs[0]
                if dat.get(locs[0] - 1) == "-":
                    rep = locs[0]
                elif dat.get(locs[-1] + 1) == "-":
                    rep = locs[-1]
                for l in locs:
                    if (
                        dat.get(l + 1j) == "|"
                        or dat.get(l - 1j) == "|"
                        or dat.get(l - 1j - 1) == "\\"
                        or dat.get(l + 1j + 1) == "\\"
                    ):
                        rep = l
                name = "".join(
                    literal_dat[v + c.imag * 1j] for v in range(int(c.real), int(r.real))
                )
                if name.startswith("@@@"):
                    name = name[3:]
                    volume_depth_start_points[l] = 0
                if name.startswith("@@2@"):
                    name = name[4:]
                    volume_depth_start_points[l] = 2
                if name.startswith("@@4@"):
                    name = name[4:]
                    volume_depth_start_points[l] = 4
                if name.startswith("@@5@"):
                    name = name[4:]
                    volume_depth_start_points[l] = 5
                if name.startswith("@@6@"):
                    name = name[4:]
                    volume_depth_start_points[l] = 6
                if name.startswith("@@8@"):
                    name = name[4:]
                    volume_depth_start_points[l] = 8
                for l in locs:
                    named[l] = name
                    c2rep[l] = rep

        def rep_neighbors(origin: complex) -> Iterator[complex]:
            if origin != c2rep.get(origin):
                return
            center = origin
            while center in literal_dat:
                center -= 1
            center += 1
            while center in literal_dat:
                for traveller, direction, _ in local_neighbors(center):
                    while True:
                        if traveller in literal_dat:
                            break
                        (traveller,) = [
                            start2
                            for start2, delta2, _ in local_neighbors(traveller)
                            if delta2 == direction
                        ]
                    if c2rep[traveller] != origin:
                        yield c2rep[traveller]
                center += 1

        # Solve for 3d location of position.
        flat2vol = {}
        for pos, c in sorted(
            dat.items(),
            key=lambda e: (e[0] not in volume_depth_start_points, e[0].real, e[0].imag, e[1]),
        ):
            if pos not in flat2vol and list(local_neighbors(pos)):
                d = volume_depth_start_points.get(pos, 0)
                stack: list[tuple[complex, Pos]] = [(pos, Pos(xy=pos - d * (1 + 1j), t=d))]
                while stack:
                    pos2, pos3 = stack.pop()
                    if pos2 in flat2vol:
                        assert flat2vol[pos2] == pos3
                        continue
                    flat2vol[pos2] = pos3
                    for pos2b, _, dif_pos3b in local_neighbors(pos2):
                        stack.append((pos2b, pos3 + dif_pos3b))

        nodes: list[tuple[str, Pos]] = []
        v2p = {v: k for k, v in flat2vol.items()}
        for pos in dat.keys():
            if pos == c2rep.get(pos):
                nodes.append((named[pos], flat2vol[pos]))
        nodes = sorted(nodes, key=lambda e: e[1])
        p2i = {v2p[v]: i for i, (_, v) in enumerate(nodes)}

        edges: list[tuple[int, int]] = []
        for pos in dat.keys():
            for f in rep_neighbors(pos):
                a = p2i[pos]
                b = p2i[f]
                if a > b:
                    a, b = b, a
                edges.append((a, b))
        edges = sorted(set(edges))

        return ZXLatticeGraph(nodes=tuple(nodes), edges=tuple(edges)).after_shrink()

    def to_basis_normals(self) -> tuple[dict[tuple[Pos, Pos], Pos], dict[tuple[int, int], Pos]]:
        node_normals: collections.defaultdict[Pos, dict[Pos, Literal["X", "Z"]]] = (
            collections.defaultdict(dict)
        )
        edge_normals: collections.defaultdict[tuple[int, int], dict[Pos, Literal["X", "Z"]]] = (
            collections.defaultdict(dict)
        )
        p2ds: collections.defaultdict[Pos, list[Pos]] = collections.defaultdict(list)
        pd2e: collections.defaultdict[Pos, dict[Pos, tuple[int, int]]] = collections.defaultdict(
            dict
        )
        for k1, k2 in self.edges:
            p1 = self.nodes[k1][1]
            p2 = self.nodes[k2][1]
            u = (p2 - p1).unit()
            p2ds[p1].append(u)
            p2ds[p2].append(-u)
            pd2e[p1][u] = (k1, k2)
            pd2e[p2][-u] = (k1, k2)

        @dataclasses.dataclass(frozen=True, unsafe_hash=True)
        class _ColorHints:
            node_pos: Pos
            node_normal: Pos
            basis: Literal["X", "Z"]

        @dataclasses.dataclass(frozen=True, unsafe_hash=True)
        class _NodeEdgeConstraint:
            node_pos: Pos
            node_normal: Pos
            edge_dir: Pos
            edge_normal: Pos
            differs: bool

        @dataclasses.dataclass(frozen=True, unsafe_hash=True)
        class _EdgeEdgeConstraint:
            edge_node1: Pos
            edge_dir1: Pos
            edge_normal1: Pos
            edge_node2: Pos
            edge_dir2: Pos
            edge_normal2: Pos
            differs: bool

        @dataclasses.dataclass(frozen=True, unsafe_hash=True)
        class _NodeNodeConstraint:
            node_pos_1: Pos
            node_normal_1: Pos
            node_pos_2: Pos
            node_normal_2: Pos
            differs: bool

        hints: set[_ColorHints] = set()
        node_edge_constraints: set[_NodeEdgeConstraint] = set()
        edge_edge_constraints: set[_EdgeEdgeConstraint] = set()
        node_node_constraints: set[_NodeNodeConstraint] = set()
        skipped_normals = set()
        dx = Pos(xy=1, t=0)
        dy = Pos(xy=1j, t=0)
        dt = Pos(xy=0, t=1)
        for k, (name, pos) in enumerate(self.nodes):
            ds = p2ds[pos]
            if name == "H":
                if len(ds) != 2:
                    raise NotImplementedError(f"{name=} {len(ds)=}")
                d1, d2 = ds
                d3 = d1.cross(d2)
                if d3:
                    for sign in [-1, +1]:
                        edge_edge_constraints.add(
                            _EdgeEdgeConstraint(
                                edge_node1=pos,
                                edge_dir1=d1,
                                edge_normal1=d3 * sign,
                                edge_node2=pos,
                                edge_dir2=d2,
                                edge_normal2=d3 * sign,
                                differs=True,
                            )
                        )
                        edge_edge_constraints.add(
                            _EdgeEdgeConstraint(
                                edge_node1=pos,
                                edge_dir1=d2,
                                edge_normal1=d1,
                                edge_node2=pos,
                                edge_dir2=d1,
                                edge_normal2=d2 * sign,
                                differs=True,
                            )
                        )
                else:
                    n1 = d1.cross(dx) or d1.cross(dy)
                    n2 = d1.cross(n1)
                    for n in [n1, n2, -n1, -n2]:
                        edge_edge_constraints.add(
                            _EdgeEdgeConstraint(
                                edge_node1=pos,
                                edge_dir1=d1,
                                edge_normal1=n,
                                edge_node2=pos,
                                edge_dir2=d2,
                                edge_normal2=n,
                                differs=True,
                            )
                        )
            elif name == "X" or name == "Z":
                for d in dx, dy, dt:
                    if d not in ds:
                        node_node_constraints.add(
                            _NodeNodeConstraint(
                                node_pos_1=pos,
                                node_pos_2=pos,
                                node_normal_1=d,
                                node_normal_2=-d,
                                differs=False,
                            )
                        )
                for k2 in range(len(ds) - 1):
                    d1 = ds[k2]
                    d2 = ds[k2 + 1]
                    d3 = d1.cross(d2)
                    if len(ds) == 2 and not d3:
                        d3 = dx.cross(d1) or dy.cross(d1)
                    if d3:
                        for s in [-1, +1]:
                            if len(ds) > 2:
                                node_normals[pos][d3 * s] = cast(Any, name)
                            else:
                                hints.add(
                                    _ColorHints(
                                        node_pos=pos, node_normal=d3 * s, basis=cast(Any, name)
                                    )
                                )

                        for d4 in [dx, dy, dt, -dx, -dy, -dy]:
                            if d4 == d3:
                                continue
                            node_node_constraints.add(
                                _NodeNodeConstraint(
                                    node_pos_1=pos,
                                    node_pos_2=pos,
                                    node_normal_1=d3,
                                    node_normal_2=d4,
                                    differs=bool(d3.cross(d4)),
                                )
                            )
                for de in ds:
                    for d2 in [dx, dy, dt, -dx, -dy, -dy]:
                        d_normal = de.cross(d2)
                        if d_normal in ds or not d_normal:
                            continue

                        node_edge_constraints.add(
                            _NodeEdgeConstraint(
                                node_pos=pos,
                                node_normal=d_normal,
                                edge_dir=de,
                                edge_normal=d_normal,
                                differs=False,
                            )
                        )
                        node_edge_constraints.add(
                            _NodeEdgeConstraint(
                                node_pos=pos,
                                node_normal=d_normal,
                                edge_dir=de,
                                edge_normal=d_normal.cross(de),
                                differs=True,
                            )
                        )
                        node_edge_constraints.add(
                            _NodeEdgeConstraint(
                                node_pos=pos,
                                node_normal=d_normal,
                                edge_dir=de,
                                edge_normal=-d_normal.cross(de),
                                differs=True,
                            )
                        )
                for d in ds:
                    skipped_normals.add((pos, d))
                if len(ds) == 1:
                    other = "Z" if name == "X" else "X"
                    node_normals[pos][-ds[0]] = cast(Any, other)

        changes = True
        while changes:
            changes = False
            for c in node_node_constraints:
                b1 = c.node_normal_1 in node_normals[c.node_pos_1]
                b2 = c.node_normal_2 in node_normals[c.node_pos_2]
                if b1 and not b2:
                    val = node_normals[c.node_pos_1][c.node_normal_1]
                    if c.differs:
                        val = "X" if val == "Z" else "Z"
                    changes = True
                    node_normals[c.node_pos_2][c.node_normal_2] = val
                if b2 and not b1:
                    val = node_normals[c.node_pos_2][c.node_normal_2]
                    if c.differs:
                        val = "X" if val == "Z" else "Z"
                    changes = True
                    node_normals[c.node_pos_1][c.node_normal_1] = val
            for c in node_edge_constraints:
                e = pd2e[c.node_pos][c.edge_dir]
                val1 = node_normals[c.node_pos].get(c.node_normal)
                val2 = edge_normals[e].get(c.edge_normal)
                if val1 is not None and val2 is not None:
                    if (val1 != val2) != c.differs:
                        raise ValueError(f"Clash: {c}")
                if val1 is not None and val2 is None:
                    val2 = val1
                    if c.differs:
                        val2 = "X" if val1 == "Z" else "Z"
                    changes = True
                    edge_normals[e][c.edge_normal] = val2
                if val2 is not None and val1 is None:
                    val1 = val2
                    if c.differs:
                        val1 = "X" if val2 == "Z" else "Z"
                    changes = True
                    node_normals[c.node_pos][c.node_normal] = val1
            for c in edge_edge_constraints:
                e1 = pd2e[c.edge_node1][c.edge_dir1]
                e2 = pd2e[c.edge_node2][c.edge_dir2]
                b1 = c.edge_normal1 in edge_normals[e1]
                b2 = c.edge_normal2 in edge_normals[e2]
                if b1 and not b2:
                    val = edge_normals[e1][c.edge_normal1]
                    if c.differs:
                        val = "X" if val == "Z" else "Z"
                    changes = True
                    edge_normals[e2][c.edge_normal2] = val
                if b2 and not b1:
                    val = edge_normals[e2][c.edge_normal2]
                    if c.differs:
                        val = "X" if val == "Z" else "Z"
                    changes = True
                    edge_normals[e1][c.edge_normal1] = val
            if not changes:
                for c in hints:
                    val = node_normals[c.node_pos].get(c.node_normal)
                    if val is None:
                        node_normals[c.node_pos][c.node_normal] = c.basis
                        changes = True
                        break

        for pos, d in skipped_normals:
            node_normals[pos].pop(d, None)
        for name, pos in self.nodes:
            if name == "Y":
                for d in [-dx, -dy, -dt, dx, dy, dt]:
                    node_normals[pos][d] = "Y"

        return dict(node_normals), dict(edge_normals)

    def x_widths(self, diam: float) -> dict[float, tuple[float, float]]:
        h2names = collections.defaultdict(set)
        for name, pos in self.nodes:
            h2names[pos.xy.real].add(name)

        start = int(min(h2names.keys())) - 10
        stop = int(max(h2names.keys())) + 10
        center_of_prev = -10
        result = {}
        for row in range(start, stop):
            if h2names[row] == {"H"}:
                center_of_prev += 0.02
                result[row] = (center_of_prev - 0.02, center_of_prev + 0.02)
                center_of_prev += 0.02
            else:
                center_of_prev += 0.5
                result[row] = (center_of_prev - diam / 2, center_of_prev + diam / 2)
                center_of_prev += 0.5
        return result

    def y_heights(self, diam: float) -> dict[float, tuple[float, float]]:
        h2names = collections.defaultdict(set)
        for name, pos in self.nodes:
            h2names[pos.xy.imag].add(name)

        start = int(min(h2names.keys())) - 10
        stop = int(max(h2names.keys())) + 10
        center_of_prev = -10
        result = {}
        for row in range(start, stop):
            if h2names[row] == {"H"}:
                center_of_prev += 0.02
                result[row] = (center_of_prev - 0.02, center_of_prev + 0.02)
                center_of_prev += 0.02
            else:
                center_of_prev += 0.5
                result[row] = (center_of_prev - diam / 2, center_of_prev + diam / 2)
                center_of_prev += 0.5
        return result

    def _to_3d_model_data(
        self, *, wireframe: bool, diam: float, text_scale: float
    ) -> tuple[list[gen.ColoredTriangleData], list[gen.ColoredLineData], list[gen.TextData]]:
        triangles: list[gen.ColoredTriangleData] = []
        lines: list[gen.ColoredLineData] = []
        text: list[gen.TextData] = []
        x_widths = self.x_widths(diam=diam)
        y_heights = self.y_heights(diam=diam)

        p2ds = collections.defaultdict(list)
        for k1, k2 in self.edges:
            p1 = self.nodes[k1][1]
            p2 = self.nodes[k2][1]
            u = (p2 - p1).unit()
            p2ds[p1].append(u)
            p2ds[p2].append(-u)

        node_normals, edge_normals = self.to_basis_normals()

        def dproj(pos: Pos, d: Pos = Pos(0, 0)) -> np.ndarray:
            x1, x2 = x_widths[pos.xy.real]
            y1, y2 = y_heights[pos.xy.imag]
            xc = (x2 + x1) / 2
            yc = (y2 + y1) / 2
            x = xc + (x2 - x1) * d.xy.real / 2
            y = yc + (y2 - y1) * d.xy.imag / 2
            t = pos.t + diam * (d.t / 2)
            return np.array([-x, -y, -t], dtype=np.float32)

        for name, pos in self.nodes:
            if name == "X" or name == "Z" or name == "Y" or name == "H":
                rgba = {
                    "X": (1, 0, 0, 1),
                    "Z": (0, 0, 1, 1),
                    "Y": (0, 1, 0, 1),
                    "H": (0.75, 0.75, 0, 1),
                }[name]
                if wireframe or name == "H":
                    triangles.append(
                        gen.ColoredTriangleData(
                            rgba=rgba,
                            triangle_list=np.array(
                                [
                                    dproj(pos, Pos(-1 - 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                    dproj(pos, Pos(-1 - 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(+1 + 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                ],
                                dtype=np.float32,
                            ),
                        )
                    )
                else:
                    lines.append(
                        gen.ColoredLineData(
                            rgba=(0, 0, 0, 1),
                            edge_list=np.array(
                                [
                                    dproj(pos, Pos(-1 - 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, -1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(+1 + 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                    dproj(pos, Pos(+1 + 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, -1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(+1 + 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, -1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(+1 - 1j, +1)),
                                    dproj(pos, Pos(-1 - 1j, +1)),
                                    dproj(pos, Pos(-1 + 1j, +1)),
                                ],
                                dtype=np.float32,
                            ),
                        )
                    )
                    for k, v in node_normals[pos].items():
                        u1 = k.triconj()
                        u2 = k.triconj().triconj()
                        rgba = {"X": (1, 0, 0, 1), "Z": (0, 0, 1, 1)}[v]
                        triangles.append(
                            gen.ColoredTriangleData(
                                rgba=rgba,
                                triangle_list=np.array(
                                    [
                                        dproj(pos, -u1 - u2 + k),
                                        dproj(pos, +u1 - u2 + k),
                                        dproj(pos, -u1 + u2 + k),
                                        dproj(pos, +u1 + u2 + k),
                                        dproj(pos, +u1 - u2 + k),
                                        dproj(pos, -u1 + u2 + k),
                                    ],
                                    dtype=np.float32,
                                ),
                            )
                        )
            else:
                assert len(p2ds[pos]) == 1, f"Unrecognized node name {name=} isn't a leaf port."
                (d,) = p2ds[pos]
                flip = False
                if d == Pos(xy=1, t=0):
                    right = (-0.1, 0, 0)
                    up = (0, 0.1, 0)
                    flip = True
                elif d == Pos(xy=0, t=1):
                    right = (0, 0, -0.1)
                    up = (0, 0.1, 0)
                    flip = True
                elif d == Pos(xy=0, t=-1):
                    right = (0, 0, -0.1)
                    up = (0, 0.1, 0)
                elif d == Pos(xy=-1j, t=0):
                    right = (0, -0.1, 0)
                    up = (0, 0, -0.1)
                elif d == Pos(xy=1j, t=0):
                    right = (0, -0.1, 0)
                    up = (-0.1, 0, 0)
                    flip = True
                else:
                    right = (-0.1, 0, 0)
                    up = (0, 0.1, 0)
                right = np.array(right, dtype=np.float32)
                up = np.array(up, dtype=np.float32)
                right *= text_scale
                up *= text_scale
                text.append(
                    gen.TextData(
                        text=name,
                        start=dproj(pos) - up * 0.5,
                        forward=right * (-1 if flip else +1),
                        up=up,
                    )
                )

        for ke, (k1, k2) in enumerate(self.edges):
            name1, pos1 = self.nodes[k1]
            name2, pos2 = self.nodes[k2]
            if wireframe or (k1, k2) not in edge_normals:
                lines.append(
                    gen.ColoredLineData(
                        rgba=(0, 0, 0, 0), edge_list=np.array([dproj(pos1), dproj(pos2)])
                    )
                )
            else:
                for normal, basis in edge_normals[(k1, k2)].items():
                    rgba = (1, 0, 0, 1) if basis == "X" else (0, 0, 1, 1)
                    d1 = (pos2 - pos1).unit()
                    d2 = normal.cross(d1)
                    triangles.append(
                        gen.ColoredTriangleData(
                            rgba=rgba,
                            triangle_list=np.array(
                                [
                                    dproj(pos1, normal + d1 - d2),
                                    dproj(pos2, normal - d1 - d2),
                                    dproj(pos1, normal + d1 + d2),
                                    dproj(pos2, normal - d1 + d2),
                                    dproj(pos2, normal - d1 - d2),
                                    dproj(pos1, normal + d1 + d2),
                                ],
                                dtype=np.float32,
                            ),
                        )
                    )

        return triangles, lines, text

    def to_3d_model(
        self, *, wireframe: bool = False, diam: float = 0.2, text_scale: float = 4
    ) -> pygltflib.GLTF2:
        triangles, lines, text = self._to_3d_model_data(
            wireframe=wireframe, text_scale=text_scale, diam=diam
        )
        return gen.gltf_model_from_colored_triangle_data(
            triangles, colored_line_data=lines, text_data=text
        )


def assert_graph_implements_permutation(
    graph: ZXLatticeGraph,
    func: Callable[[dict[str, int]], dict[str, int]],
    *,
    inputs: list[str] | None = None,
):
    if inputs is None:
        inputs = [
            name for name, pos in graph.nodes if name.endswith("_in") or name.startswith("in_")
        ]
    order = sorted(
        name
        for (name, pos) in graph.nodes
        if not name.endswith(">") and name not in ["X", "Z", "H"]
    )
    if len(order) != len(set(order)):
        raise ValueError(f"order isn't unique: {order=}")
    actual = graph.to_tensor(port_order_key=lambda name, _: order.index(name))
    n = len(order)
    expected = np.zeros(shape=(2,) * n, dtype=np.complex64)
    assert expected.shape == actual.shape, "shape"
    num_in = len(inputs)
    seen = set(np.flatnonzero(actual))
    for ks in list(seen):
        kvs1 = {order[len(order) - k - 1]: (ks >> k) & 1 for k in range(len(order))}
        kvs3 = func(kvs1)
        if kvs3.pop("--ignore--", False):
            seen.remove(ks)
            continue
        kvs2 = {**kvs1, **kvs3}
        assert kvs1.keys() == kvs2.keys()
        if kvs1 != kvs2:
            lines = ["Differences:"]
            for k in sorted(kvs1.keys()):
                v1 = kvs1[k]
                v2 = kvs2[k]
                if v1 != v2:
                    lines.append(f"    {v1} {v2} {k} <<<<<<<<<<<")
                else:
                    lines.append(f"    {v1} {v2} {k}")
            assert False, "\n".join(lines)

        assert kvs1 == kvs2
    for vals in itertools.product([0, 1], repeat=num_in):
        ins = {k: v for k, v in zip(inputs, vals)}
        outs = func(ins)
        if outs.pop("--ignore--", False):
            continue
        index: list[int | None] = [None] * n
        for k, v in {**ins, **outs}.items():
            index[order.index(k)] = int(v)
        if None in index:
            raise ValueError(
                f"Missing a port assignment: {set(order) - (outs.keys() | ins.keys())}"
            )
        expected[tuple(index)] = 2 ** -(num_in / 2)
    allowed = set(np.flatnonzero(expected))
    if allowed != seen:
        lines = []
        lines.append("Produced incorrect outputs that shouldn't be possible.")
        for k in allowed - seen:
            lines.append(f"Missing: {k}")
        for k in seen - allowed:
            lines.append(f"Extra: {k}")
        raise ValueError("\n".join(lines))
