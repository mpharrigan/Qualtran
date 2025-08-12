SCATTER SCRIPT
==============

This directory contains a simple simulator for fuzzing reversible classical logic circuits that contain a small number
of quantum tweaks. The simulator is specialized for mostly-classical circuits, and works by tracking several
randomly chosen values through the circuit while tracking relative phases.

When the simulator is created, a number of branches is chosen.
When the simulator allocates a register, the register can be initialized to a constant value (like 0) or to a
"scattered value" (the stand-in for a superposition).
A scattered value will have a randomly chosen value within each branch.
Registers can be operated on, and then assertions can be made about invariants that should be true for all branches.
For example:

```python
from scatter_script import QPU
qpu = QPU(num_branches=32)

register1 = qpu.alloc_quint(length=10, scatter=True)
register2 = qpu.alloc_quint(length=10, scatter=True)
register3 = qpu.alloc_quint(length=10, val=0)
register3 += register1
register3 ^= -1  # apply a NOT gate to all bits in the register
register3 += register2
register3 ^= -1
assert register3 == (register1 - register2) % 2**10
```

In addition to classical arithmetic, the simulator supports the following quantum operations:

- Phase gates (Z, S, T, CZ, etc):
    The simulator tracks a phase value for each branch. Phase gates adjust these values. Generally
    it's expected that the phases should be zero'd by the end of the computation. This can be asserted
    by calling `qpu.verify_clean_finish()`.
- Measurement based uncomputation (`register.del_measure_x()` and `register.del_measure_qft()`):
    A value that is a function of other allocated values can be instantly deleted by measuring it in
    a basis orthogonal to the computational basis. This perturbs the phases of states, with the measurement
    result indicating the perturbation that must be corrected.

For example, an AND gate can be uncomputed without a Toffoli via `del_measure_x`:

```python
from scatter_script import QPU

qpu = QPU(num_branches=32)

a = qpu.alloc_quint(length=1, scatter=True)
b = qpu.alloc_quint(length=1, scatter=True)
c = qpu.alloc_quint(length=1, val=0)
c ^= a & b  # Compute with Toffoli

# Measurement based uncomputation
mx = c.del_measure_x()
if mx:
    qpu.cz(a, b)  # Phase correction.

a.UNPHYSICAL_force_del(dealloc=True)
b.UNPHYSICAL_force_del(dealloc=True)
qpu.verify_clean_finish()
```

WHAT THE SIMULATOR CAN'T VERIFY
===============================

The simulator cannot verify that measurement based uncomputations are only performed on values that are functions of
other remaining values. Nothing will stop you from doing the following:

```python
from scatter_script import QPU
qpu = QPU(num_branches=32)

# An example of an impossible uncomputation that the simulator will not catch as a mistake.
a = qpu.alloc_quint(length=1, scatter=True)
unphysical_rvalue_copy = a + 0
if a.del_measure_x():
    qpu.z(unphysical_rvalue_copy)
```

More generally, if a value is removed by measurement based uncomputation and it isn't a computational basis function of
other remaining values then what *should* happen is various complex interference effects. The simulator has no mechanism
for detecting that this is occurring, or of correctly implementing those interference effects, so instead it produces a
silently incorrect result. For similar reasons, the simulator can't verify that you didn't measure an input you weren't
supposed to.

In general, the scatter script simulator is useful for verifying that complex reversible logic has been
correctly implemented.
It's secondarily useful for verifying that well-intentioned complex measurement
based uncomputations don't contain various silly mistakes.
Don't rely on it for correctness beyond that.
