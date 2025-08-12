from __future__ import annotations

import random
from typing import Any

import numpy as np

from facto.operations.zx_lattice_graph import ZXLatticeGraph, assert_graph_implements_permutation
from scatter_script import quint, QPU

test_graph = r"""
             ccx0  ccx1  ccx2                                            
               |     |     |                                            
 a0_in------Z--|--Z--|--Z--|---a0_out                                      
             \ |     |   \ |                                            
              \|     |    \|                                            
               X     X     X                                            
               |     |     |                                                                
 s0_in------Z--|--Z--|--Z--|---s0_out                                     
             \ |   \ |   \ |                                          
              \|    \|    \|                                       
               X     X     X                                      
                     |     |                                                                
 b0_in------Z-----Z--|--Z--|---b0_out                                     
                   \ |   \ |                                          
                    \|    \|                                       
                     X     X                      ccx0b ccx1b ccx2b                                    
                           |                        |     |     |                                                                
 a1_in------Z-----Z-----Z--|---------------------Z--|--Z--|--Z--|---a1_out                                      
                         \ |                      \ |     |   \ |                                          
                          \|                       \|     |    \|                                       
                           X                        X     X     X                                      
                           |                        |     |     |                                                                
                        Z--|---------------------Z--|--Z--|--Z--|---s1_out                                    
                         \ |                      \ |   \ |   \ |                                          
                          \|                       \|    \|    \|                                       
                           X                        X     X     X                                      
                           |                              |     |                                                                
 b1_in------Z-----Z-----Z--|---------------------Z-----Z--|--Z--|---b1_out                                      
                         \ |                            \ |   \ |                                          
                          \|                             \|    \|                                       
                           X                              X     X                                      
                                                                |                                                                                                        
                               a2_in-------------Z-----Z-----Z--|---a2_out                                                                                                
                                                              \ |                                                                                                        
                                                               \|                                                                                                        
                                                                X                                                                                                        
                                                                |                                                                                                        
                                                             Z--|---s2_out                                                                                                
                                                              \ |                                                                                                        
                                                               \|                                                                                                        
                                                                X                                                                                                        
                                                                |                                                                                                        
                               b2_in-------------Z-----Z-----Z--|---b2_out                                                                                                
                                                              \ |                                                                                                        
                                                               \|                                                                                                        
                                                                X                                                                                                        
"""


test_graph_2 = r"""
                                                                                                                    
                                                                                                                    
                         ccx0   ccx1   ccx2                           ccx0b  ccx1b  ccx2b                                                                                     
                            \      \     |                               \      \     |                                                                                   
      a0_in                  \      \    |        a1_in                   \      \    |          a2_in                                     
          \                   X      X   |           \                     X      X   |             \                                   
           \                  |      |   |            \                    |      |   |              \                                 
     Z------Z--------------Z--|---Z--|---X-------------Z----------------Z--|---Z--|---X---------------Z------Z                                
             \              \ |      |   |              \                \ |      |   |                \                              
              \              \|      |   |               \                \|      |   |                 \                             
             a0_out           X      X   |             a1_out              X      X   |               a2_out                          
                              |      |   |                                 |      |   |                                                                   
   s0_in----Z--------------Z--|---Z--|---X-------------Z----------------Z--|---Z--|---X---------------Z------Z                         
             \              \ |    \ |   |              \                \ |    \ |   |                \                            
       b0_in  \              \|     \|   |         b1_in \                \|     \|   |           b2_in \                        
          \    s0_out         X      X   |           \    s1_out           X      X   |             \    s2_out                 
           \                         |   |            \                           |   |              \                                                    
     Z------Z---------------------Z--|---X-------------Z-----------------------Z--|---X---------------Z------Z                               
             \                     \ |                  \                       \ |                    \                            
              \                     \|                   \                       \|                     \                        
             b0_out                  X                 b1_out                     X                   b2_out                              
"""


def test_to_tensor():
    g = ZXLatticeGraph.from_text(
        """
        a---Z---b
            |
            c
    """
    )
    np.testing.assert_allclose(
        g.to_tensor().reshape((8,)), np.array([0.5**0.5, 0, 0, 0, 0, 0, 0, 0.5**0.5])
    )

    g = ZXLatticeGraph.from_text(
        """
        a---X---b
            |
            c
    """
    )
    np.testing.assert_allclose(
        g.to_tensor().reshape((8,)), np.array([1, 0, 0, 1, 0, 1, 1, 0]) * 0.5
    )

    g = ZXLatticeGraph.from_text(
        """
        a---H---b
    """
    )
    np.testing.assert_allclose(g.to_tensor(), np.array([[1, 1], [1, -1]]) * 0.5)

    g = ZXLatticeGraph.from_text(
        """
        a---Z---b
            |
            H---c
    """
    )
    np.testing.assert_allclose(g.to_tensor().reshape(8), np.array([1, 1, 0, 0, 0, 0, 1, -1]) * 0.5)

    g = ZXLatticeGraph.from_text(
        """
        c0---Z---c1
             |
        t0---X---t1
    """
    )
    np.testing.assert_allclose(
        g.to_tensor().reshape((4, 4)),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) * 0.5,
    )

    g = ZXLatticeGraph.from_text(
        """
        in---Z---Z---out
             |   |
             Z---Z 
    """
    )
    np.testing.assert_allclose(g.to_tensor().reshape((2, 2)), np.array([[1, 0], [0, 1]]) * 0.5**0.5)

    g = ZXLatticeGraph.from_text(
        """
        |ccx[0]>----a

        |ccx[1]>----b

        |ccx[2]>----c
    """
    )
    np.testing.assert_allclose(
        g.to_tensor().reshape((4, 2)), np.array([[1, 0], [1, 0], [1, 0], [0, 1]]) * 0.5
    )

    g = ZXLatticeGraph.from_text(
        """
        a0----Z----a1
              |
           |ccx[0]>

        b0----Z----b1
              |
           |ccx[1]>

        c0----X----c1
              |
           |ccx[2]>
    """
    )
    np.testing.assert_allclose(
        g.to_tensor().reshape((8, 8)),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )
        / 2**1.5,
    )

    g = ZXLatticeGraph.from_text(
        """
        a0----Z-------Z----a1
              |       |
          |ccx[0]>  |ccxB[0]>

        b0----Z-------Z----b1
              |       |
          |ccx[1]>  |ccxB[1]>

        c0----X-------X----c1
              |       |
          |ccx[2]>  |ccxB[2]>
    """
    )
    np.testing.assert_allclose(
        g.to_tensor().reshape((8, 8)),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        / 2**1.5,
    )


def test_add_graph():
    g1 = ZXLatticeGraph.from_text(test_graph)
    g2 = ZXLatticeGraph.from_text(test_graph_2)
    f1 = g1.to_function()
    f2 = g2.to_function()
    assert f1 == f2


def add_init(a: quint, b: quint, *, out: quint, qc: QPU, uncompute_key: Any = "add") -> None:
    for k in range(len(a)):
        a0 = a[k - 1] if k else 0
        b0 = b[k - 1] if k else 0
        s0 = out[k - 1] if k else 0
        a1 = a[k]
        b1 = b[k]

        cmpa, cmpb, _ = qc.alloc_wandering_and(a0 ^ s0, b0 ^ s0, out=out[k])
        out[k] ^= a0 ^ b0 ^ a1 ^ b1 ^ s0
        if cmpa:
            out[k] ^= b0 ^ s0
        if cmpb:
            out[k] ^= a0 ^ s0
        if cmpa & cmpb:
            out[k] ^= 1

        # Leftover phase on a, b, and a^b
        phase_as = random.random() < 0.5
        phase_bs = random.random() < 0.5
        phase_ab = random.random() < 0.5
        if phase_as:
            qc.s_dag(a0 ^ s0)
        if phase_bs:
            qc.s_dag(b0 ^ s0)
        if phase_ab:
            qc.s_dag(a0 ^ b0)
        qc.push_uncompute_info(phase_as, key=uncompute_key)
        qc.push_uncompute_info(phase_bs, key=uncompute_key)
        qc.push_uncompute_info(phase_ab, key=uncompute_key)


def add_del(a: quint, b: quint, *, out: quint, qc: QPU, uncompute_key: Any = "add") -> None:
    for k in range(len(a))[::-1]:
        a0 = a[k - 1] if k else 0
        b0 = b[k - 1] if k else 0
        s0 = out[k - 1] if k else 0
        a1 = a[k]
        b1 = b[k]

        m = out[k].del_measure_x()
        phase_ab = qc.pop_uncompute_info(key=uncompute_key)
        phase_bs = qc.pop_uncompute_info(key=uncompute_key)
        phase_as = qc.pop_uncompute_info(key=uncompute_key)

        # S corrections
        if phase_ab ^ m:
            qc.s(a0 ^ b0)
        if m ^ phase_as:
            qc.s(a0 ^ s0)
        if m ^ phase_bs:
            qc.s(b0 ^ s0)

        # Z corrections
        if m & phase_bs:
            qc.z(b0 ^ s0)
        if phase_ab & m:
            qc.z(a0 ^ b0)
        if m & phase_as:
            qc.z(a0 ^ s0)
        if m:
            qc.z(a1 ^ b1 ^ s0)


def test_add_raw():
    for _ in range(100):
        n = 10
        qc = QPU(num_branches=10)
        a = qc.alloc_quint(length=n, scatter=True)
        b = qc.alloc_quint(length=n, scatter=True)
        c = qc.alloc_quint(length=n)
        a0 = a.UNPHYSICAL_copy()
        b0 = b.UNPHYSICAL_copy()
        add_init(a, b, out=c, qc=qc)
        assert a0 == a
        assert b0 == b
        assert (a0 + b0) % 2**n == c
        add_del(a, b, out=c, qc=qc)
        a.UNPHYSICAL_force_del(dealloc=True)
        b.UNPHYSICAL_force_del(dealloc=True)
        qc.verify_clean_finish()


def test_adder_build_1():
    g = ZXLatticeGraph.from_text(
        r"""
                                    |ccx[0]>
                                      |
                                      X  |ccx[1]> X
                                      |\   |      |\
            a0_in----Z------------------Z-----------Z---a0_out
                     |                |    |      |            
                     |                X    X      X            
                     |                 \   |\     |\             
                     X------------------Z----Z------Z---s0_out
                     |                     |      |             
                     |                     X      X             
                     |                      \     |\            
            b0_in----Z-----------------------Z------Z---b0_out
                                                  |            
                                                  X            
                                                  |\            
            a1_in-----------------------------------Z---a1_out
                                                  |             
                                      |ccx[2]>    X
                                         \        |\            
                                          X---------X---s1_out
                                                  |
                                                  X             
                                                   \
            b1_in-----------------------------------Z---b1_out
    """
    )

    def func(kv: dict[str, int]) -> dict[str, int]:
        a0_in = kv["a0_in"]
        b0_in = kv["b0_in"]
        a1_in = kv["a1_in"]
        b1_in = kv["b1_in"]
        return {
            "a0_out": a0_in,
            "b0_out": b0_in,
            "s0_out": a0_in ^ b0_in,
            "a1_out": a1_in,
            "b1_out": b1_in,
            "s1_out": ((a0_in + b0_in + 2 * a1_in + 2 * b1_in) >> 1) & 1,
        }

    assert_graph_implements_permutation(g, func, inputs=["a0_in", "b0_in", "a1_in", "b1_in"])


def test_adder_build_2():
    g = ZXLatticeGraph.from_text(
        r"""
                      |ccx[0]> |ccx[1]>        
                           |    |       
                           |    |               |ccx[2]>
                           |    |                   |
                           X    X                   |
                           |\   |                   |
                     Z-------Z-----------Z----------X----a1_in                    
                     |     |    |         \         |             
                     |     X    X         a0_in     |          
                     |      \   |\                  |              
                     X-------Z----Z------Z----------X----s1_out
                     |          |         \         |              
                     |          X         s0_out    |                 
                     |           \                  |
                     Z------------Z------Z----------X----b1_in             
                                          \          
                                           b0_in         
    """
    )

    def func(kv: dict[str, int]) -> dict[str, int]:
        a0_in = kv["a0_in"]
        b0_in = kv["b0_in"]
        a1_in = kv["a1_in"]
        b1_in = kv["b1_in"]
        s = a0_in + b0_in + 2 * a1_in + 2 * b1_in
        return {"s0_out": s & 1, "s1_out": bool(s & 2)}

    assert_graph_implements_permutation(g, func, inputs=["a0_in", "b0_in", "a1_in", "b1_in"])


def test_adder_build_3():
    g = ZXLatticeGraph.from_text(
        r"""
                       |ccxA[0]>      |ccxA[1]>                |ccxB[0]>        |ccxB[1]>
                           |           |                           |              |
                           |           |          |ccxA[2]>        |              |           |ccxB[2]>
                           |           |               |           |              |                |
                           X           X               |           X              X                |
                           |\          |\              |           |\             |\               |
                     X-------Z-----------Z-------------X-------------Z--------------Z--------------X----Z
                     |     |  \        |               |           |  \           |                |     \
                     |     X s0_out    X               |           X   s1_out     X                |      s2_out
                     |      \          |               |            \             |                |
                     X-------Z-----------Z-------------X-------------Z--------------Z--------------X----Z
                     |        \        |               |              \           |                |     \
                     |       a0_z      X               |               a1_z       X                |      a2_z 
                     |                  \              |                           \               |
                     X-------Z-----------Z-------------X-------------Z--------------Z--------------X----Z
                              \                                       \                                  \
                               b0_z                                    b1_z                               b2_z 
    """
    )

    def func(kv: dict[str, int]) -> dict[str, int]:
        a0_z = kv["a0_z"]
        b0_z = kv["b0_z"]
        a1_z = kv["a1_z"]
        b1_z = kv["b1_z"]
        a2_z = kv["a2_z"]
        b2_z = kv["b2_z"]
        s = a0_z + b0_z + 2 * a1_z + 2 * b1_z + 4 * a2_z + 4 * b2_z
        return {"s0_out": s & 1, "s1_out": bool(s & 2), "s2_out": bool(s & 4)}

    assert_graph_implements_permutation(
        g, func, inputs=["a0_z", "b0_z", "a1_z", "b1_z", "a2_z", "b2_z"]
    )
    assert g.to_3d_model() is not None


def test_tof_teleport():
    for _ in range(20):
        qc = QPU(num_branches=20)
        val = qc.alloc_quint(length=3, scatter=True)
        ent = qc.alloc_quint(length=3)
        ent ^= val
        tof = qc.alloc_toffoli_state()
        a, b, t = val
        tof[0] ^= a
        tof[1] ^= b
        t ^= tof[2]
        ma = tof[0].UNPHYSICAL_copy()
        mb = tof[1].UNPHYSICAL_copy()
        tof[0].UNPHYSICAL_force_del(dealloc=True)
        tof[1].UNPHYSICAL_force_del(dealloc=True)
        mxt = tof[2].del_measure_x()
        if mxt:
            qc.z(ma & mb)
            qc.z(ma & b)
            qc.z(a & mb)
            qc.z(a & b)
        t ^= b & ma
        t ^= a & mb
        t ^= ma & mb

        ent[2] ^= ent[0] & ent[1]
        assert ent == val
        ent.UNPHYSICAL_force_del(dealloc=True)
        val.UNPHYSICAL_force_del(dealloc=True)
        qc.verify_clean_finish()
