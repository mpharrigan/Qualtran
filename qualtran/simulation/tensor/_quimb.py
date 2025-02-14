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
from typing import Any, cast, Dict, Iterable, Tuple

import attrs
import numpy as np
import quimb.tensor as qtn

from qualtran import (
    Bloq,
    CompositeBloq,
    Connection,
    LeftDangle,
    QBit,
    Register,
    RightDangle,
    Soquet,
    SoquetT,
)
from qualtran._infra.composite_bloq import _cxns_to_cxn_dict, BloqBuilder

logger = logging.getLogger(__name__)


def cbloq_to_quimb(cbloq: CompositeBloq, friendly_indices: bool = False) -> qtn.TensorNetwork:
    """Convert a composite bloq into a tensor network.

    This function will call `Bloq.my_tensors` on each subbloq in the composite bloq to add
    tensors to a quimb tensor network. This method has no default fallback, so you likely want to
    call `bloq.as_composite_bloq().flatten()` to decompose-and-flatten all bloqs down to their
    smallest form first. The small bloqs that result from a flattening 1) likely already have
    their `my_tensors` method implemented; and 2) can enable a more efficient tensor contraction
    path.

    Args:
        cbloq: The composite bloq
        friendly_indices: If set to True, the outer indices of the tensor network will be renamed
            from their Qualtran-computer-readable form to human-friendly strings. This may be
            useful if you plan on manually manipulating the resulting tensor network but will
            preclude any further processing by Qualtran functions.

    Returns:
        The tensor network
    """
    tn = qtn.TensorNetwork([])

    logging.info(
        "Constructing a tensor network for composite bloq of size %d", len(cbloq.bloq_instances)
    )

    for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
        bloq = binst.bloq
        assert isinstance(bloq, Bloq)

        inc_d = _cxns_to_cxn_dict(bloq.signature.lefts(), pred_cxns, get_me=lambda cxn: cxn.right)
        out_d = _cxns_to_cxn_dict(bloq.signature.rights(), succ_cxns, get_me=lambda cxn: cxn.left)

        for tensor in bloq.my_tensors(inc_d, out_d):
            if isinstance(tensor, DiscardInd):
                # TODO finish error message
                raise ValueError(
                    f"During tensor simulation, {bloq} tried to discard information. This requires using TODO-open-system-sim-func"
                )
            tn.add(tensor)

    # Special case: Add variables corresponding to all registers that don't connect to any Bloq.
    # This is needed because `CompositeBloq.iter_bloqnections` ignores `LeftDangle/RightDangle`
    # bloqs, and therefore we never see connections that exist only b/w LeftDangle and
    # RightDangle bloqs.
    for cxn in cbloq.connections:
        if cxn.left.binst is LeftDangle and cxn.right.binst is RightDangle:
            # This register has no Bloq acting on it, and thus it would not have a variable in
            # the tensor network. Add an identity tensor acting on this register to make sure the
            # tensor network has variables corresponding to all input / output registers.

            for j in range(cxn.left.reg.bitsize):
                placeholder = Soquet(None, Register('simulation_placeholder', QBit()))  # type: ignore
                Connection(cxn.left, placeholder)
                tn.add(
                    qtn.Tensor(
                        data=np.eye(2),
                        inds=[
                            (Connection(cxn.left, placeholder), j),
                            (Connection(placeholder, cxn.right), j),
                        ],
                    )
                )

    if friendly_indices:
        return tn.reindex(_get_friendly_indices(tn))
    return tn


def _get_friendly_indices(tn: 'qtn.TensorNetwork') -> Dict[Any, str]:
    """Go through a tensor network's outer inds to map them to unique strings."""
    ind_name_map: Dict[Any, str] = {}

    # Each index is a (cxn: Connection, j: int) tuple.
    cxn: Connection
    j: int

    for ind in tn.outer_inds():
        cxn, j = ind
        if cxn.left.binst is LeftDangle:
            soq = cxn.left
            side = 'l'
        elif cxn.right.binst is RightDangle:
            soq = cxn.right
            side = 'r'
        else:
            raise ValueError(f"Unknown side for {cxn}")

        idx_str = f'{soq.idx}' if soq.idx else ''
        ind_name_map[ind] = f'{soq.reg.name}{idx_str}_{j}{side}'
    return ind_name_map


@attrs.frozen
class DiscardInd:
    ind_tuple: Tuple[str, int]


def make_forward_tensor(t: qtn.Tensor):
    new_inds = [(*ind, True) for ind in t.inds]

    t2 = t.copy()
    t2.modify(inds=new_inds)
    return t2


def make_backward_tensor(t: qtn.Tensor):
    new_inds = []
    for ind in t.inds:
        new_inds.append((*ind, False))

    t2 = t.H
    t2.modify(inds=new_inds, tags=t.tags | {'dag'})
    return t2


def cbloq_to_superquimb(cbloq: CompositeBloq, friendly_indices: bool = False) -> qtn.TensorNetwork:
    """Convert a composite bloq into a tensor network.

    This function will call `Bloq.my_tensors` on each subbloq in the composite bloq to add
    tensors to a quimb tensor network. This method has no default fallback, so you likely want to
    call `bloq.as_composite_bloq().flatten()` to decompose-and-flatten all bloqs down to their
    smallest form first. The small bloqs that result from a flattening 1) likely already have
    their `my_tensors` method implemented; and 2) can enable a more efficient tensor contraction
    path.
    """
    tn = qtn.TensorNetwork([])

    logging.info(
        "Constructing a tensor network for composite bloq of size %d", len(cbloq.bloq_instances)
    )

    for binst, pred_cxns, succ_cxns in cbloq.iter_bloqnections():
        bloq = binst.bloq
        assert isinstance(bloq, Bloq)

        inc_d = _cxns_to_cxn_dict(bloq.signature.lefts(), pred_cxns, get_me=lambda cxn: cxn.right)
        out_d = _cxns_to_cxn_dict(bloq.signature.rights(), succ_cxns, get_me=lambda cxn: cxn.left)

        for tensor in bloq.my_tensors(inc_d, out_d):
            if isinstance(tensor, DiscardInd):
                dind = tensor.ind_tuple
                tn.reindex({(*dind, True): dind, (*dind, False): dind}, inplace=True)
            else:
                forward_tensor = make_forward_tensor(tensor)
                backward_tensor = make_backward_tensor(tensor)
                tn.add(forward_tensor)
                tn.add(backward_tensor)

    if friendly_indices:
        tn = tn.reindex(_get_friendly_superindices(tn))
    return tn


def _get_friendly_superindices(tn: 'qtn.TensorNetwork') -> Dict[Any, str]:

    ind_name_map: Dict[Any, str] = {}
    for ind in tn.outer_inds():
        cxn, j, forward = ind
        if cxn.left.binst is LeftDangle:
            soq = cxn.left
            side = 'l'
        elif cxn.right.binst is RightDangle:
            soq = cxn.right
            side = 'r'
        else:
            raise ValueError(f"Unknown side for {cxn}")

        d = 'f' if forward else 'b'
        idx = f'{soq.idx}' if soq.idx else ''
        ind_name_map[ind] = f'{soq.reg.name}{idx}_{j}{side}{d}'

    return ind_name_map


def _add_classical_kets(bb: BloqBuilder, registers: Iterable[Register]) -> Dict[str, 'SoquetT']:
    """Use `bb` to add `IntState(0)` for all the `vals`."""

    from qualtran.bloqs.basic_gates import IntState

    soqs: Dict[str, 'SoquetT'] = {}
    for reg in registers:
        if reg.shape:
            reg_vals = np.zeros(reg.shape, dtype=int)
            soq = np.empty(reg.shape, dtype=object)
            for idx in reg.all_idxs():
                soq[idx] = bb.add(IntState(val=cast(int, reg_vals[idx]), bitsize=reg.bitsize))
        else:
            soq = bb.add(IntState(val=0, bitsize=reg.bitsize))

        soqs[reg.name] = soq
    return soqs


def initialize_from_zero(bloq: Bloq):
    """Take `bloq` and compose it with initial zero states for each left register.

    This can be contracted to a state vector for a given unitary.
    """
    bb = BloqBuilder()

    # Add the initial 'kets' according to the provided values.
    in_soqs = _add_classical_kets(bb, bloq.signature.lefts())

    # Add the bloq itself
    out_soqs = bb.add_d(bloq, **in_soqs)
    return bb.finalize(**out_soqs)
