#!/usr/bin/env python3

import gen

from util_gen_stub_file import generate_documentation


def main():
    objects = [
        obj
        for obj in generate_documentation(obj=gen, full_name="gen", level=0)
        if all("[DEPRECATED]" not in line for line in obj.lines)
    ]

    print(f"# gen v{gen.__version__} API Reference")
    print()
    print("## Index")
    for obj in objects:
        level = obj.level
        print((level - 1) * "    " + f"- [`{obj.full_name}`](#{obj.full_name})")

    print(
        f"""
```python
# Types used by the method definitions.
from typing import overload, TYPE_CHECKING, Any, Iterable
import io
import pathlib
import numpy as np
```
""".strip()
    )

    for obj in objects:
        print()
        print(f'<a name="{obj.full_name}"></a>')
        print("```python")
        print(f"# {obj.full_name}")
        print()
        if len(obj.full_name.split(".")) > 2:
            print(f'# (in class {".".join(obj.full_name.split(".")[:-1])})')
        else:
            print(f"# (at top-level in the gen module)")
        print("\n".join(obj.lines))
        print("```")


if __name__ == "__main__":
    main()
