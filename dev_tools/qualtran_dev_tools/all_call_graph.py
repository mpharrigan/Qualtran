#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""The complete, library-wide call graph."""

from typing import Iterable, Optional, Type

import cirq
import networkx as nx
import pydot
import sympy

from qualtran import (
    Adjoint,
    Bloq,
    BloqExample,
    Controlled,
    DecomposeNotImplementedError,
    DecomposeTypeError,
)
from qualtran.bloqs.basic_gates import (
    IntState,
    Rx,
    Ry,
    Rz,
    TGate,
    Toffoli,
    XPowGate,
    YPowGate,
    ZPowGate,
)
from qualtran.cirq_interop import CirqGateAsBloq
from qualtran.drawing.bloq_counts_graph import GraphvizCounts
from qualtran.resource_counting import SympySymbolAllocator
from qualtran.resource_counting.bloq_counts import _build_call_graph, _make_composite_generalizer
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    generalize_cvs,
    generalize_rotation_angle,
    ignore_alloc_free,
    ignore_cliffords,
    ignore_split_join,
)

# TODO: Make a function that figures out the true leaf nodes: Something is a leaf node
#       if all of its children are in the target gateset. Could we just pop out the target
#       gateset nodes? I want to fold them upwards.


def get_all_call_graph(bes: Iterable[BloqExample]):
    generalize = _make_composite_generalizer(
        cirq_to_bloqs,
        ignore_split_join,
        generalize_cvs,
        generalize_rotation_angle,
        ignore_alloc_free,
        ignore_cliffords,
    )

    def keep(b: Bloq) -> bool:
        if isinstance(b, IntState):
            if isinstance(b.bitsize, sympy.Expr):
                return True  # bug 1
            if (not isinstance(b.bitsize, sympy.Expr)) and b.bitsize > 64:
                return True  # bug 2

        if b == Toffoli():
            return True

        if b == TGate():
            return True

        if isinstance(b, (Rx, Ry, Rz, XPowGate, YPowGate, ZPowGate)):
            return True

        if isinstance(b, CirqGateAsBloq) and isinstance(b.gate, cirq.ControlledGate):
            # wacky
            return True

        if isinstance(b, CirqGateAsBloq):
            # Causes cycles in the class graph
            return True

        try:
            children = [bloq for bloq, n in b.build_call_graph(SympySymbolAllocator())]
        except DecomposeTypeError:
            return True
        except DecomposeNotImplementedError:
            print(f"Warning: {b} decomposition not implemented.")
            return True

        return False

    g = nx.DiGraph()
    ssa = SympySymbolAllocator()

    for be in bes:
        bloq = be.make()
        _build_call_graph(
            bloq=bloq, generalizer=generalize, ssa=ssa, keep=keep, max_depth=None, g=g, depth=0
        )

    return g


def call_graph_to_class_graph(g: nx.Graph):
    """Take a call graph and turn it into a class graph.

    In a call graph, the nodes are `Bloq`s and edges contain a count.
    Here, the edges are a bloq _class_, and all the different instantiations
    of it in the original call graph are collapsed into one node. We drop edge attributes.

    An edge exists from caller class to callee class if _any_ of the instantiations depend
    on the callee class.
    """
    g2 = nx.DiGraph()

    def bloq_to_node(b: Bloq) -> Type[Bloq]:
        if isinstance(b, Adjoint):
            return b.subbloq.__class__
        if isinstance(b, Controlled):
            return b.subbloq.__class__
        if isinstance(b, CirqGateAsBloq):
            return b.gate.__class__
        return b.__class__

    for b in g.nodes:
        g2.add_node(bloq_to_node(b))

    for b1, b2 in g.edges:
        g2.add_edge(bloq_to_node(b1), bloq_to_node(b2))

    return g2


class ClassGraphGraphviz(GraphvizCounts):
    """Drawer that supports the class graph from `call_graph_to_class_graph`."""

    def get_graph(self):
        """Get the pydot graph."""
        graph = pydot.Dot('classes', graph_type='digraph', rankdir='TB', ranksep=0.8)
        self.add_nodes(graph)
        self.add_edges(graph)
        return graph

    def add_edges(self, graph):
        for b1, b2 in self.g.edges:
            graph.add_edge(pydot.Edge(self.get_id(b1), self.get_id(b2)))

    def get_node_title(self, b):
        return b.__name__

    def get_node_details(self, b):
        return ''
