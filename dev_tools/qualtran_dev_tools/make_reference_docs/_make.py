#  Copyright 2024 Google LLC
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
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import cast, List, Sequence, Dict, Tuple

# Begin monkeypatch:
# the `Visitor` will use certain decorators to apply "labels" to the AST nodes, which
# causes properties to be parsed as Attributes instead of Functions. By removing these
# decorator-to-label mappings, they are kept as Function. We handle decorators for functions
# in this script how we want.
import _griffe.agents.visitor
import attrs
import griffe
from griffe import (
    DocstringSection,
    DocstringSectionAdmonition,
    DocstringSectionKind,
    DocstringSectionParameters,
    DocstringSectionReturns,
    DocstringSectionText,
    GriffeLoader,
    Kind,
)

del _griffe.agents.visitor.builtin_decorators['property']
del _griffe.agents.visitor.stdlib_decorators['functools.cached_property']

PARSER = 'google'


class LinkingWriter:
    def __init__(self, f):
        self._f = f
        self._linked = set()
        ...

    def repl(self, ma: re.Match):
        l = ma.group(1)
        self._linked.add(l)
        return f'[`{l}`]'

    def linkify(self, s: str):
        return re.sub(r'`(qualtran\.[\w\.]+)`', self.repl, s)

    def write(self, s: str):
        self._f.write(self.linkify(s))

    def write_nl(self, s: str):
        self._f.write(s)

    def write_link_targets(self):
        self._f.write('\n')
        for href in self._linked:
            # TODO: pref_path to rel_path
            # TODO: rename pref_path to pref_dotname or something
            trg_segments = href.split('.')
            trg = '/'.join(trg_segments[:-1]) + f'/{trg_segments[-1]}.md'
            self._f.write(f'[`{href}`]: {trg}\n')


def write_docstring_parts(f, parts: List[DocstringSection], level: int):
    lvl = '#' * level
    for part in parts:
        if part.kind is DocstringSectionKind.text:
            f.write(part.value.strip())
            f.write('\n\n')

        elif part.kind is DocstringSectionKind.parameters:
            part = cast(DocstringSectionParameters, part)
            f.write(f'###{lvl} Args\n')
            for param in part.value:
                # f.write(f'{param.name=} {param.value=} {param.annotation=} {param.description=}')
                if param.annotation:
                    f.write(f'{param.name}: `{param.annotation}`\n')
                else:
                    f.write(f'{param.name}\n')
                f.write(f': {param.description}')  # myst syntax for definition lists in markdown
                f.write('\n\n')

        elif part.kind is DocstringSectionKind.returns:
            part = cast(DocstringSectionReturns, part)
            f.write(f'###{lvl} Returns\n')

            # One, unnamed return value
            if len(part.value) == 1 and not part.value[0].name:
                f.write(part.value[0].description)
                f.write('\n')

            # Multiple return values and/or named return values
            else:
                for param in part.value:
                    # f.write(f'{param.name=} {param.value=} {param.annotation=} {param.description=}')
                    pname = param.name if param.name else 'ret'
                    f.write(f'{pname}\n')
                    f.write(
                        f': {param.description}'
                    )  # myst syntax for definition lists in markdown
                    f.write('\n\n')

        elif part.kind is DocstringSectionKind.admonition:
            part = cast(DocstringSectionAdmonition, part)
            part = part.value
            if part.kind == 'see-also':
                f.write(f'###{lvl} See Also\n')
                f.write(part.description)
                f.write('\n\n')
            else:
                warnings.warn(f"Unknown admonition type {part.kind}")

        else:
            f.write(str(part))
            f.write('\n')
            warnings.warn(f"Unknown docstring {part}")
    f.write('\n\n')


def format_parameter(p: griffe.Parameter) -> str:
    if p.annotation is not None and p.default is not None:
        s = f"{p.name}: {p.annotation} = {p.default}"
    elif p.annotation is not None and p.default is None:
        s = f"{p.name}: {p.annotation}"
    elif p.annotation is None and p.default is not None:
        s = f"{p.name} = {p.default}"
    else:
        s = f"{p.name}"

    if p.kind and p.kind is griffe.ParameterKind.positional_or_keyword:
        return s
    elif p.kind and p.kind is griffe.ParameterKind.positional_only:
        return s  # TODO?
    elif p.kind and p.kind is griffe.ParameterKind.keyword_only:
        return s  # TODO?
    elif p.kind and p.kind is griffe.ParameterKind.var_positional:
        return f'*{s}'
    elif p.kind and p.kind is griffe.ParameterKind.var_keyword:
        return f'**{s}'
    else:
        raise ValueError(p.kind)


def write_property_method_signature(f, obj, obj2: griffe.Function):
    assert obj2.returns, obj2
    obj_instance_name = camel_to_snake(obj.name)
    method_signature = f'{obj_instance_name}.{obj2.name} -> {obj2.returns}'
    f.write(f'```python\n{method_signature}\n```\n\n')


def camel_to_snake(name):
    # Turn an UpperCamelCaseName into lower_snake_case
    # .. we want to make it look like we're calling methods on instances of the class.
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return s1.lower()


def get_obj_instance_name(obj):
    # Get prototypical object instance names for a given class.

    # First, check if we have a bespoke name
    n = {'BloqBuilder': 'bb'}.get(obj.name, None)
    if n is not None:
        return n

    # Otherwise, use the class name as lower_snake_case.
    return camel_to_snake(obj.name)


def write_generic_method_signature(
        f, obj, obj2: griffe.Function, first_arg_name='self', caller_name=get_obj_instance_name
):
    # Strip `self` or `cls` argument.
    parameters = list(obj2.parameters)
    p0 = parameters[0]
    assert p0.name == first_arg_name, p0  # method
    parameters = parameters[1:]

    if len(parameters) > 1:
        # One line
        params = '\n' + ',\n'.join(f'  {format_parameter(p)}' for p in parameters) + '\n'
    else:
        # Multiline
        params = ', '.join(format_parameter(p) for p in parameters)

    # For methods, make it look like we're calling it on an instance.
    caller_name = caller_name(obj)

    # All together
    if obj2.returns is not None:
        method_signature = f'{caller_name}.{obj2.name}({params}) -> {obj2.returns}'
    else:
        method_signature = f'{caller_name}.{obj2.name}({params})'
    f.write(f'```python\n{method_signature}\n```\n\n')


def write_method_signature(f, obj, obj2: griffe.Function):
    if obj2.overloads:
        # todo .. handle overlaods
        warnings.warn(f'{obj.name}.{obj2.name} has {len(obj2.overloads)} overloads')

    # Dispatch to different writers based on decorators
    if obj2.decorators:

        # Filter out decorators that don't matter.
        decs = [d for d in obj2.decorators if str(d.value) not in ['abc.abstractmethod']]

        if len(decs) == 1:
            (d,) = decs

            # It's a property, write a special property signature.
            if str(d.value) == 'property' or str(d.value) == 'cached_property':
                return write_property_method_signature(f, obj, obj2)

            # It's a classmethod, write a special classmethod signature.
            elif str(d.value) == 'classmethod':
                return write_generic_method_signature(
                    f, obj, obj2, first_arg_name='cls', caller_name=lambda obj: obj.name
                )

            # Fallback: just skip it.
            else:
                # todo .. handle other decorators
                warnings.warn(f'{obj.name}.{obj2.name} has decorator {d.value!r}')
                return
        else:
            # todo .. handle more than one decorator.
            warnings.warn(f'{obj.name}.{obj2.name} has multiple decorators.')
            return

    # No decorators, write a normal method signature.
    return write_generic_method_signature(f, obj, obj2)


def _get_first_line(obj):
    if obj.docstring is None:
        return ''
    dp0, *dparts = obj.docstring.parse(PARSER)
    if dp0.kind is DocstringSectionKind.text:
        first_line, *other_lines = re.split(r'\n{2,}', dp0.value, flags=re.MULTILINE)
        return first_line

    else:
        warnings.warn(f"Unknown first part in {obj}")
        return ''


def _get_writers_for_split_docstring(obj: griffe.Object, level=0):
    """Extract the first, summary line of a docstring from an object.

    This returns two closures that take an `f` to write to. The first writes the first
    line and the second writes the rest. Both must be called.
    """

    if obj.docstring is None:
        def first_part(f):
            return

        def second_part(f):
            f.write('\n\n')

        return first_part, second_part

    dp0, *dparts = obj.docstring.parse(PARSER)
    if dp0.kind is DocstringSectionKind.text:
        first_line, *other_lines = re.split(r'\n{2,}', dp0.value, flags=re.MULTILINE)

        def first_part(f):
            f.write(first_line)
            f.write('\n\n')

        dp0 = DocstringSectionText('\n\n'.join(other_lines))
    else:
        warnings.warn(f"Unknown first part in {obj}")

        def first_part(f):
            return

    def second_part(f):
        write_docstring_parts(f, [dp0] + dparts, level=level)

    return first_part, second_part


def write_major_class(f, obj: griffe.Class):
    # Title
    f.write(f"# {obj.name}\n")

    # Class docstring
    d0, drest = _get_writers_for_split_docstring(obj, level=0)
    d0(f)
    f.write('## Overview\n')  # Annoyingly have to include an <h2> before <h3> or sphinx freaks out
    drest(f)

    for name, obj2 in obj.members.items():
        if obj2.is_private:
            continue
        if obj2.name == '__init__':
            continue
        if obj2.is_special:
            # TODO: do we want these?
            continue
        if obj2.kind is Kind.ATTRIBUTE:
            # TODO: do we want these?
            continue

        # Member name
        f.write(f'## `{obj2.name}`\n')

        # First docstring line
        d0, drest = _get_writers_for_split_docstring(obj2, level=0)
        d0(f)

        # Optional: signature
        if obj2.kind is Kind.FUNCTION:
            write_method_signature(f, obj, cast(griffe.Function, obj2))

        # Rest of docstring
        drest(f)


def write_module(f, obj: griffe.Module, members):
    # Title
    f.write(f'# {obj.name}\n\n')

    d0, drest = _get_writers_for_split_docstring(obj, level=0)
    d0(f)
    f.write('## Overview\n')  # Annoyingly have to include an <h2> before <h3> or sphinx freaks out
    drest(f)

    f.write('\n## Modules\n')
    for obj, pref_path, mytype in members:
        if mytype != 'module':
            continue

        summ = _get_first_line(obj)
        f.write(f'`{pref_path}`: {summ}\n\n')

    f.write('\n## Major Classes\n')
    for obj, pref_path, mytype in members:
        if mytype != 'major':
            continue

        summ = _get_first_line(obj)
        f.write(f'`{pref_path}`: {summ}\n\n')

    f.write('\n## Other Members\n')
    for obj, pref_path, mytype in members:
        if mytype in ['major', 'module']:
            continue

        d0, drest = _get_writers_for_split_docstring(obj, level=1)
        f.write_nl(f'### `{pref_path}`\n')
        d0(f)
        drest(f)

        submemb: griffe.Object
        doc_submembers = {(submemb_name, submemb) for submemb_name, submemb in obj.members.items()
                          if (not submemb.is_special) and submemb.has_docstring}
        if doc_submembers:
            f.write('#### Members\n')
            for submemb_name, submemb in doc_submembers:
                subdesc = _get_first_line(submemb)
                f.write_nl(f'`{submemb_name}`\n')
                f.write(f': {subdesc}\n\n')

        if obj.inherited_members:
            f.write('**All Members:** ')
            f.write(
                ', '.join(f'`{submemb_name}`' for submemb_name, submemb in obj.all_members.items()
                          if (not submemb.is_special) and submemb.is_public))
            f.write('\n')


def render_major_class(base_dir: Path, pref_path: str, obj: griffe.Class | griffe.Alias):
    segments = pref_path.split('.')
    out_path = base_dir / '/'.join(segments[:-1]) / f'{segments[-1]}.md'
    print(f"Writing {pref_path} to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        write_major_class(f, obj)


def render_module(base_dir: Path, pref_path: str, obj: griffe.Module | griffe.Alias, members):
    segments = pref_path.split('.')
    out_path = base_dir / '/'.join(segments[:-1]) / f'{segments[-1]}.md'
    print(f"Writing {pref_path} to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        f2 = LinkingWriter(f)
        write_module(f2, obj, members)
        f2.write_link_targets()


@attrs.frozen
class PageLoc:
    page_name: str
    section_name: str
    preferred_name: str
    other_names: Sequence[str]


@attrs.mutable
class Page:
    obj: griffe.Module = None
    kind: str = None
    section: str = None
    pref_path: str = None
    # obj, pref_path, mytype
    members: List[Tuple[griffe.Object, str, str]] = attrs.field(factory=list)


MAJOR_CLASSES = [
    'qualtran.Bloq',
    'qualtran.CompositeBloq',
    'qualtran.BloqBuilder',
    'qualtran.Signature',
    'qualtran.dtype.QCDType',
    'qualtran.resource_counting.QECGatesCost',
    'qualtran.resource_counting.GateCounts',
    'qualtran.resource_counting.QubitCount',
    'qualtran.resource_counting.CostKey',
    'qualtran.resource_counting.SuccessProb',
    'qualtran.resource_counting.BloqCount',
    'qualtran.simulation.classical_sim.ClassicalSimState',
    'qualtran.simulation.classical_sim.PhasedClassicalSimState',
]

SKIP_MODULES = ['qualtran.conftest', 'qualtran.testing_test',
                'qualtran.protos', 'qualtran.serialization.resolver_dict',
                'qualtran.bloqs']


def walk_table_of_contents(obj: griffe.Object, toc, mod_pages):
    if obj.is_module and obj.canonical_path not in SKIP_MODULES:
        for name, obj2 in obj.members.items():
            if obj2.kind is Kind.ALIAS:
                # print(obj2.path)
                continue

            if obj2.canonical_path in toc:
                # Already found it
                continue

            if obj2.is_private:
                continue

            if obj2.is_special:
                continue

            if obj2.is_module and obj2.name.endswith('_test'):
                # print(obj2.path)
                continue

            if obj2.is_module and obj2.name.endswith('_pb2'):
                # print(obj2.path)
                continue

            walk_table_of_contents(obj2, toc, mod_pages)
            assert obj2.canonical_path in toc

    # Need to find the "container" for `obj`.
    # Look at its aliases, find the shortest, find its parents.
    if len(obj.aliases) == 0:
        all_aliases = set()
    else:
        all_aliases = set(obj.aliases.keys())

    all_aliases.add(obj.path)
    all_aliases.add(obj.canonical_path)
    defined_container = '.'.join(obj.canonical_path.split('.')[:-1])

    def prefered_path_key(alias: str):
        if alias.startswith('qualtran.dtype'):
            # Special case for qualtran.dtype re-exports. Doesn't follow any of the other rules.
            return -1
        if alias.startswith('qualtran.exception'):
            # Special case for qualtran.dtype re-exports. Doesn't follow any of the other rules.
            return -1

        alias_container = '.'.join(alias.split('.')[:-1])
        if not defined_container.startswith(alias_container):
            return float('inf')

        return len(alias.split('.'))

    pref_path = sorted(all_aliases, key=prefered_path_key)[0]
    pref_parent = '.'.join(pref_path.split('.')[:-1])

    if obj.is_module:
        mod_pages[pref_parent].members.append((obj, pref_path, 'module'))

        mod_pages[pref_path].obj = obj
        mod_pages[pref_path].kind = 'module'
        mod_pages[pref_path].pref_path = pref_path

        mod_pages[pref_path].section = '.'.join(pref_path.split('.')[:2])

        toc[obj.canonical_path] = PageLoc(
            page_name=pref_path,  # .. todo
            section_name='',  # .. todo
            preferred_name=pref_path,
            other_names=tuple(all_aliases - {pref_path})
        )
    elif pref_path in MAJOR_CLASSES:
        mod_pages[pref_parent].members.append((obj, pref_path, 'major'))

        mod_pages[pref_path].obj = obj
        mod_pages[pref_path].kind = 'class'
        mod_pages[pref_path].section = '.'.join(pref_parent.split('.')[:2])
        mod_pages[pref_path].pref_path = pref_path

        toc[obj.canonical_path] = PageLoc(
            page_name=None,
            section_name='',  # .. todo
            preferred_name=pref_path,
            other_names=tuple(all_aliases - {pref_path})
        )
    else:
        mod_pages[pref_parent].members.append((obj, pref_path, 'minor'))
        toc[obj.canonical_path] = PageLoc(
            page_name=None,
            section_name='',  # .. todo
            preferred_name=pref_path,
            other_names=tuple(all_aliases - {pref_path})
        )


def make_reference_docs(p: Path):
    out_dir = p / 'docs/reference'

    loader = GriffeLoader()
    mod = loader.load("qualtran")

    unresolved, _ = loader.resolve_aliases()
    assert len(unresolved) == 0
    assert mod.is_module
    assert mod.is_init_module
    assert mod.parent is None

    # Mapping of canonical name to page name
    toc: Dict[str, PageLoc] = {}
    pages = defaultdict(Page)
    walk_table_of_contents(mod, toc, pages)

    pages = [p for p in pages.values() if p.obj is not None]  # .. todo

    # gtoc = defaultdict(lambda: defaultdict(list))
    # for canon, pg in toc.items():
    #     assert canon.startswith('qualtran.')
    #     canon = canon[len('qualtran.'):]
    #
    #     gtoc[pg.section_name][pg.page_name].append(pg)
    # gtoc = dict(gtoc)

    top_sections = [
        'qualtran',
        'qualtran.resource_counting',
        'qualtran.simulation',
        'qualtran.drawing',
        'qualtran.symbolics',
    ]
    fake_sections = [
        # Grouped in `qualtran`.
        'qualtran.dtype',
        'qualtran.exception',
    ]
    sections = top_sections + sorted(
        set(mp.section for mp in pages) - set(top_sections) - set(fake_sections))

    def _pages_sort_key(p: Page):
        path_parts = p.pref_path.split('.')
        return path_parts

    def _page_in_section(p: Page, section: str):
        if p.section == section:
            return True
        if section == 'qualtran' and p.section in ['qualtran.dtype', 'qualtran.exception']:
            # Group
            return True
        return False

    with (out_dir / 'toc.md').open('w') as f:
        print("Writing toc.md ..")
        f.write('# Qualtran\n\n')

        for section in sections:
            if section == '':
                section = 'Base'
            f.write(f'## `{section}`\n\n')

            spages: List[Page] = sorted((p for p in pages if _page_in_section(p, section)),
                                        key=_pages_sort_key)

            for mp in spages:
                if mp.kind == 'module' and len(mp.members) == 0:
                    print(f"Skipping empty module page {mp.pref_path}")
                    continue

                f.write(f'### {mp.pref_path}\n')
                f.write(f'kind: {mp.kind}\n')
                for item, item_pp, typ in mp.members:
                    if typ == 'major':
                        pass
                    else:
                        f.write(f' - {item_pp} ({typ})\n')

            f.write('\n')

    for page in pages:
        if page.pref_path in MAJOR_CLASSES:
            render_major_class(out_dir, page.pref_path, page.obj)
        else:
            render_module(out_dir, page.pref_path, page.obj, page.members)
