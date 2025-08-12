from __future__ import annotations

import argparse
import pathlib

import gen
from facto.operations.zx_lattice_graph import ZXLatticeGraph, assert_graph_implements_permutation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True, type=str)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    g1 = ZXLatticeGraph.from_text(
        r"""
                        |ccx[0]> |ccx[1]>   |ccx[2]>                                        
                            |     |           |                             
            s0_in--------Z--|--Z--|--------Z--|-------------------------s0_out                                     
                          \ |   \ |         \ |                                                                
                           \|    \|          \|                                                             
                            X     X           X                                                            
                            |     |           |                                                                                      
            a0_in--------Z--|-----|--------Z--|-------------------------a0_out                                      
                          \ |     |         \ |                                                                  
                           \|     |          \|                                                                  
                            X     |           X                                                                  
                                  |           |                                                                                      
            b0_in--------------Z--|--------Z--|-------------------------b0_out                                     
                                \ |         \ |                           
                                 \|          \|                        
                                  X           X                                                            
                                              |                                                                                      
                                           Z--|-------------------------s1_out                                    
                                            \ |                                                                
                                             \|                                                             
                                              X                                                            
                                              |                                                                                      
            a1_in--------------------------Z--|-------------------------a1_out                                      
                                            \ |                                                                
                                             \|                                                             
                                              X                                                            
                                              |                                                                                      
            b1_in--------------------------Z--|-------------------------b1_out                                      
                                            \ |                                                                
                                             \|                                                             
                                              X                                                            
        """
    )

    def func(kv: dict[str, int]) -> dict[str, int]:
        a0_in = kv["a0_in"]
        s0_in = kv["s0_in"]
        b0_in = kv["b0_in"]
        a1_in = kv["a1_in"]
        b1_in = kv["b1_in"]
        s1_out = ((a0_in + b0_in + 2 * a1_in + 2 * b1_in) >> 1) & 1
        return {
            "a0_out": a0_in,
            "b0_out": b0_in,
            "s0_out": s0_in,
            "a1_out": a1_in,
            "b1_out": b1_in,
            "s1_out": s1_out,
            "--ignore--": s0_in != a0_in ^ b0_in,
        }

    assert_graph_implements_permutation(g1, func)
    gen.viz_3d_gltf_model_html(g1.to_3d_model(wireframe=True)).write_to(out_dir / "adder_1.html")

    g2 = ZXLatticeGraph.from_text(
        r"""
                                         |ccx[2]>
                                            |
                      |ccx[0]> |ccx[1]>     |                                         
                           |    |           |                             
                 s0_z----Z-|--Z-|-----------X-----s1_out                                                         
                          \|   \|           |                                                                
                           X    X           |                                                             
                           |    |           |                                                            
                           |    |           |                                                                                      
                 a0_z----Z-|----|-----------X-----a1_z                                                            
                          \|    |           |                                                                  
                           X    |           |                                                                  
                                |           |                                                                  
                                |           |                                                                                      
                 b0_z---------Z-|-----------X-----b1_z                                                          
                               \|                                       
                                X                                    
                                                                                                           
        """
    )

    def func(kv: dict[str, int]) -> dict[str, int]:
        a0_in = kv["a0_z"]
        s0_in = kv["s0_z"]
        b0_in = kv["b0_z"]
        a1_in = kv["a1_z"]
        b1_in = kv["b1_z"]
        s1_out = ((a0_in + b0_in + 2 * a1_in + 2 * b1_in) >> 1) & 1
        return {"s1_out": s1_out, "--ignore--": s0_in != a0_in ^ b0_in}

    assert_graph_implements_permutation(g2, func, inputs=["a0_z", "s0_z", "b0_z", "a1_z", "b1_z"])
    gen.viz_3d_gltf_model_html(g2.to_3d_model(wireframe=True)).write_to(out_dir / "adder_2.html")

    g = ZXLatticeGraph.from_text(
        r"""
                       |ccxA[0]>     |ccxA[1]>                 |ccxB[0]>       |ccxB[1]>
                           |           |                           |              |
                           |           |           |ccxA[2]>       |              |            |ccxB[2]>
                           |           |               |           |              |                |
                           X           X               |           X              X                |
                           |\          |\              |           |\             |\               |
                     X-------Z-----------Z-------------X-------------Z--------------Z--------------X----Z
                     |     |  \        |               |           |  \           |                |     \
                     |     X s0_out    X               |           X   s1_out     X                |      s2_out
                     |      \          |               |            \             |                |
                     X-------Z-------------------------X-------------Z-----------------------------X----Z
                     |        \        |               |              \           |                |     \
                     |       a0_z      X               |               a1_z       X                |      a2_z 
                     |                  \              |                           \               |
                     X-------Z-----------Z-------------X-------------Z--------------Z--------------X----Z
                              \                                       \                                  \
                               b0_z                                    b1_z                               b2_z 
    """
    )
    g2 = ZXLatticeGraph.from_text(
        r"""
                                                                                                        
                                                                                                        
                       |ccz1>      |ccz1>        |ccz1>     |ccz2>      |ccz2>        |ccz2>              
                          \           \             \          \           \             \              
                           X           X             X          X           X             X             
                           |           |              \         |           |              \            
                           |           |               X        |           |               X           
                           |           |               |        |           |               |           
                           X           X               H        X           X               H           
                           |\          |\              |        |\          |\              |           
                     X-------Z-----------Z-------------X----------Z-----------Z-------------X--------------Z
                     |     |  \        |               |        |  \        |               |               \
                     |     | s0_out    X               |        | s1_out    X               |                s2_out
                     |     |            \              |        |            \              |           
                     X-------Z-----------Z-------------X----------Z-----------Z-------------X--------------Z
                     |     |  \                        |        |  \                        |               \
                     |     X a0_z                      |        X a1_z                      |                a2_z 
                     |      \                          |         \                          |           
                     X-------Z-------------------------X----------Z-------------------------X--------------Z
                              \                                    \                                        \
                               b0_z                                 b1_z                                     b2_z
                                                                                                                        
                                                                                                                    
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

    gen.viz_3d_gltf_model_html(g.to_3d_model(wireframe=True)).write_to(out_dir / "adder_3.html")
    gen.viz_3d_gltf_model_html(g2.to_3d_model()).write_to(out_dir / "adder_4.html")
    assert_graph_implements_permutation(
        g, func, inputs=["a0_z", "b0_z", "a1_z", "b1_z", "a2_z", "b2_z"]
    )
    assert_graph_implements_permutation(
        g2, func, inputs=["a0_z", "b0_z", "a1_z", "b1_z", "a2_z", "b2_z"]
    )


if __name__ == "__main__":
    main()
