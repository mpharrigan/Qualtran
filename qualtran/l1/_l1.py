#  Copyright 2025 Google LLC
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
import inspect
import itertools
from typing import Any, cast, Dict, List, Sequence, Set, Tuple, Union

import attrs
import networkx as nx
import numpy as np

from qualtran import (
    Bloq,
    DanglingT,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    LeftDangle,
    QCDType,
    QUInt,
    RightDangle,
    Signature,
    Soquet,
)
from qualtran._infra.binst_graph_iterators import greedy_topological_sort
from qualtran._infra.composite_bloq import _binst_to_cxns, _cxns_to_soq_dict
from qualtran.bloqs.arithmetic import Add


@attrs.mutable
class Varmap:
    """Handles assigning variable names for our variable *objects*, i.e. Soquets.

    The main attribute is `varmap`, which maps `Soquet` to a unique string name
    that can be used as a variable name in a script.
    """

    varmap: Dict[Soquet, str] = attrs.field(factory=dict)
    bobjmap: Dict[Bloq, str] = attrs.field(factory=dict)
    names: Set[str] = attrs.field(factory=set)

    def __getitem__(self, k):
        return self.varmap[k]

    def unique_name(self, prefix: str) -> str:
        i = 0
        while True:
            candidate = f'{prefix}{i}'
            if candidate not in self.names:
                self.names.add(candidate)
                return candidate
            i += 1

    def assign(self, soq: Soquet, prefix: str) -> str:
        name = self.unique_name(prefix)
        self.varmap[soq] = name
        return name

    def assign_bobj(self, bloq: Bloq) -> Union[str, None]:
        if bloq in self.bobjmap:
            return None
        name = self.unique_name(prefix=str(bloq.__class__.__name__.lower()))
        self.bobjmap[bloq] = name
        return name


@attrs.frozen
class Call:
    rets: str
    ctor: str
    qargs: str


@attrs.frozen
class Input:
    varname: str
    regname: str
    dtype: QCDType


@attrs.frozen
class Output:
    varname: str
    regname: str


@attrs.frozen
class Alias:
    varname: str
    ctor: str


@attrs.mutable
class SubroutineFormatter:
    """Turn elements into code strings.

    By modifying or overriding these methods, you can customize how the code appears
    without having to dive into the compute graph logic.
    """

    me: str
    varmap: Varmap
    lines: List[str] = attrs.field(factory=list)
    calls: List[Call] = attrs.field(factory=list)
    inpts: List[Input] = attrs.field(factory=list)
    outpts: List[Output] = attrs.field(factory=list)
    aliases: List[Alias] = attrs.field(factory=list)

    def input_to_code(self, varname: str, regname: str, dtype: QCDType):
        self.inpts.append(Input(varname, regname, dtype))
        inp = f'{varname} = bb.add_register("{regname}", {dtype})'
        self.lines.append(inp)

    def add_bobj_alias(self, varname: str, bloq: Bloq):
        self.aliases.append(Alias(varname, repr(bloq)))

    def bloq_obj_to_code(self, bloq: Bloq) -> str:
        return self.varmap.bobjmap[bloq]

    def input_kwargs_to_code(self, kwargs: Sequence[Tuple[str, Any]]) -> str:
        kwargs = [f'{regname}={_arr_to_str(kwargvar)}' for regname, kwargvar in kwargs]
        kwargs = ', '.join(kwargs)
        return kwargs

    def output_rets_to_code(self, rets: Sequence[str]) -> str:
        return ', '.join(rets)

    def call_to_code(self, bloq, kwargs, rets):
        rets = self.output_rets_to_code(rets)
        ctor = self.bloq_obj_to_code(bloq)
        qargs = self.input_kwargs_to_code(kwargs)
        return f'{rets} = bb.add({ctor}, {qargs})'

    def add_call(self, bloq, kwargs, rets):
        rets = self.output_rets_to_code(rets)
        ctor = self.bloq_obj_to_code(bloq)
        qargs = self.input_kwargs_to_code(kwargs)
        self.calls.append(Call(rets, ctor, qargs))

    def flush(self) -> str:
        self.lines = []
        for call in self.calls:
            self.lines.append(f'{call.rets} = bb.add({call.ctor}, {call.qargs})')

        return '\n'.join(self.lines)

    def flush(self) -> str:
        self.lines = []

        retslen = max((len(c.rets) for c in self.calls), default=0)
        ctorlen = max((len(c.ctor) for c in self.calls), default=0)
        qargslen = max((len(c.qargs) for c in self.calls), default=0)
        aliaslen = max((len(a.varname) for a in self.aliases), default=0)
        indent = ' ' * 4

        self.lines.append(f'qdef {self.me} [')
        for inpt in self.inpts:
            self.lines.append(f'{indent}{inpt.varname}: {inpt.dtype}')
        self.lines.append('] {')

        for alias in self.aliases:
            self.lines.append(f'{indent}{alias.varname:{aliaslen}s} = {alias.ctor}')

        self.lines.append('')

        for call in self.calls:
            self.lines.append(
                f'{indent}{call.rets:{retslen}s} = {call.ctor:{ctorlen}s}[{call.qargs:{qargslen}s}]'
            )

        self.lines.append('}')

        return '\n'.join(self.lines)


def _arr_to_str(obj):
    """Turn an ndarray into a string."""
    if not isinstance(obj, np.ndarray):
        return str(obj)

    items_as_strings = [_arr_to_str(item) for item in obj]
    return f"[{', '.join(items_as_strings)}]"


def _init_args(bloq: Bloq):
    if attrs.has(type(bloq)):
        for field in attrs.fields(type(bloq)):  # type: ignore[arg-type]
            if field.name in inspect.signature(type(bloq).__init__).parameters:
                yield field


def _bobj_aliases(bloq: Bloq, varmap: Varmap, pfmt: SubroutineFormatter):
    for field in _init_args(bloq):
        subbloq = getattr(bloq, field.name)
        if isinstance(subbloq, Bloq):
            _bobj_aliases(subbloq, varmap, pfmt)
            name = varmap.assign_bobj(subbloq)
            if name is not None:
                pfmt.add_bobj_alias(name, subbloq)

    name = varmap.assign_bobj(bloq)
    if name is not None:
        pfmt.add_bobj_alias(name, bloq)


def bloq_to_code(bloq: Bloq):
    varmap = Varmap()
    pfmt = SubroutineFormatter(me=repr(bloq), varmap=varmap)

    try:
        cbloq = bloq.decompose_bloq()
    except DecomposeTypeError:
        return None, []
    except DecomposeNotImplementedError:
        return pfmt, []

    g = cbloq._binst_graph

    # Inputs
    for reg in bloq.signature.lefts():
        pfmt.input_to_code(varname=reg.name, regname=reg.name, dtype=reg.dtype)

    # Make aliases for all the bloqs we find
    for binst in greedy_topological_sort(g):
        if isinstance(binst, DanglingT):
            continue
        _bobj_aliases(binst.bloq, varmap, pfmt)

    for binst in greedy_topological_sort(g):
        preds, succs = _binst_to_cxns(binst, binst_graph=g)

        # 1. Handle input variables
        if binst is LeftDangle:
            assert len(preds) == 0
            for suc in succs:
                x = varmap.assign(suc.left, suc.left.pretty())
                reg = suc.left.reg
                if reg.shape:
                    varmap.varmap[suc.left] = f'{reg.name}[{suc.left.idx}]'
                else:
                    varmap.varmap[suc.left] = reg.name
            continue

        # 2. Handle output variables
        if binst is RightDangle:
            # TODO: add call to bb.finalize(...)
            assert len(succs) == 0
            for pred in preds:
                # print("finalize(", pred, ")")
                pass
            continue

        # 4. Input kwqargs
        inpsoqs = _cxns_to_soq_dict(
            binst.bloq.signature.lefts(),
            preds,
            get_me=lambda cxn: cxn.right,
            get_assign=lambda cxn: cxn.left,
        )
        kwargs = []
        for regname, soqs in inpsoqs.items():
            if isinstance(soqs, np.ndarray):
                arr = np.empty(soqs.shape, dtype=object)
                for idx in itertools.product(*[range(sh) for sh in soqs.shape]):
                    arr[idx] = varmap[soqs[idx]]
                kwargs.append((regname, arr))
            else:
                kwargvar = varmap[soqs]
                kwargs.append((regname, kwargvar))

        # 5. Output kwqargs
        retsoqs = _cxns_to_soq_dict(
            binst.bloq.signature.rights(),
            succs,
            get_me=lambda cxn: cxn.left,
            get_assign=lambda cxn: cxn.left,
        )
        rets = []
        for regname, soqs in retsoqs.items():
            basename = varmap.unique_name(regname)

            if isinstance(soqs, np.ndarray):
                for idx in itertools.product(*[range(sh) for sh in soqs.shape]):
                    idxn = ','.join(str(x) for x in idx)
                    varnm = f'{basename}[{idxn}]'
                    # print('adding', soqs[idx], '->', varnm)
                    varmap.varmap[cast(Soquet, soqs[idx])] = varnm
                rets.append(basename)
            else:
                varmap.varmap[soqs] = basename
                rets.append(basename)

        # 6. Emit line
        pfmt.add_call(bloq=binst.bloq, kwargs=kwargs, rets=rets)
    return pfmt, list(varmap.bobjmap.keys())


def bloqs_to_code(root: Bloq):
    fmts = []
    root_fmt, subbloqs = bloq_to_code(root)
    assert root_fmt is not None
    fmts = [root_fmt]

    while subbloqs:
        bloq = subbloqs.pop(0)
        fmt, new_subbloqs = bloq_to_code(bloq)
        subbloqs += new_subbloqs
        if fmt is not None:
            fmts.append(fmt)

    for fmt in fmts:
        print(fmt.flush())
        print()
