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
import logging
from typing import Dict, List, Set

from qualtran import Bloq, CompositeBloq, Connection, DanglingT, RightDangle, Soquet
from qualtran._infra.composite_bloq import _binst_to_cxns

logger = logging.getLogger(__name__)


def bloq_has_custom_tensors(bloq: Bloq) -> bool:
    """Whether this bloq declares custom tensors by overriding `.add_my_tensors(...)`.

    This is a heuristic that checks that the method is overriden. This is used as the
    flattening predicate in `flatten_for_tensor_contraction`.
    """
    return not bloq.add_my_tensors.__qualname__.startswith(
        'Bloq.'
    ) and not bloq.add_my_tensors.__qualname__.startswith('GateWithRegisters.')


def _remove_one_split_join_pair(cbloq: CompositeBloq):
    from qualtran.bloqs.bookkeeping import Join, Split

    left_side_cxns: Dict[int, List[Soquet]] = {}
    right_side_cxns: Dict[int, List[Soquet]] = {}
    marked: Set[int] = set()

    for binst, preds, succs in cbloq.iter_bloqnections():
        if isinstance(binst.bloq, Join):
            assert len(succs) == 1, 'Join should only have one successor'
            next_binst = succs[0].right.binst

            if next_binst is RightDangle:
                # Next binst doesn't have a bloq, continue
                continue

            if isinstance(next_binst.bloq, Split):
                # It is a join/split pair. Record the soquets and the binst i.
                left_side_cxns[binst.i] = [p.left for p in preds]
                _, next_succs = _binst_to_cxns(next_binst, cbloq._binst_graph)
                right_side_cxns[binst.i] = [s.right for s in next_succs]
                marked.add(binst.i)
                marked.add(next_binst.i)

                # Note: If it was guaranteed that we wouldn't have sequential
                # pairs of join/split pairs, we could do all of these in one fell swoop.
                # Since we can indeed have such runs of join/splits we'd have to traverse
                # the graph to get the "real" left_side_cxns and right_side_cxns. Instead,
                # we'll just remove one pair at a time.
                break

    if not marked:
        raise StopIteration()

    logger.info("Removing binsts: %s", marked)

    new_cxns: List[Connection] = []
    for cxn in cbloq.connections:
        if not isinstance(cxn.left.binst, DanglingT) and cxn.left.binst.i in marked:
            continue  # skip re-adding
        if not isinstance(cxn.right.binst, DanglingT) and cxn.right.binst.i in marked:
            continue  # skip re-adding
        new_cxns.append(cxn)
    new_binsts = frozenset(binst for binst in cbloq.bloq_instances if binst.i not in marked)

    # Connect the (removed) join's left soquets to the (removed) split's right soquets.
    for i in left_side_cxns.keys():
        for ls, rs in zip(left_side_cxns[i], right_side_cxns[i]):
            new_cxns.append(Connection(ls, rs))

    return CompositeBloq(tuple(new_cxns), cbloq.signature, new_binsts)


def remove_split_join_pairs(cbloq: CompositeBloq):
    ret = cbloq
    while True:
        try:
            ret = _remove_one_split_join_pair(ret)
        except StopIteration:
            return ret


def flatten_for_tensor_contraction(bloq: Bloq, max_depth: int = 1_000) -> CompositeBloq:
    """Flatten a (composite) bloq as much as possible to enable efficient tensor contraction.

    Without this function, bloqs without custom tensors will be contracted to a dense tensor using
    their decomposition and then that dense tensor will be used in the enclosing tensor network.
    To allow a more efficient contraction ordering, use this function to decompose-and-flatten
    as much as possible before starting the tensor contraction.
    """
    cbloq = bloq.as_composite_bloq()
    flat = cbloq.flatten(lambda binst: not bloq_has_custom_tensors(binst.bloq), max_depth=max_depth)
    flat2 = remove_split_join_pairs(flat)
    return flat2
