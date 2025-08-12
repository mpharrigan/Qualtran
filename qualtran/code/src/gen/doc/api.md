# gen v0.1.0 API Reference

## Index
- [`gen.Chunk`](#gen.Chunk)
    - [`gen.Chunk.__init__`](#gen.Chunk.__init__)
    - [`gen.Chunk.end_code`](#gen.Chunk.end_code)
    - [`gen.Chunk.end_interface`](#gen.Chunk.end_interface)
    - [`gen.Chunk.end_patch`](#gen.Chunk.end_patch)
    - [`gen.Chunk.find_distance`](#gen.Chunk.find_distance)
    - [`gen.Chunk.find_logical_error`](#gen.Chunk.find_logical_error)
    - [`gen.Chunk.flattened`](#gen.Chunk.flattened)
    - [`gen.Chunk.from_circuit_with_mpp_boundaries`](#gen.Chunk.from_circuit_with_mpp_boundaries)
    - [`gen.Chunk.start_code`](#gen.Chunk.start_code)
    - [`gen.Chunk.start_interface`](#gen.Chunk.start_interface)
    - [`gen.Chunk.start_patch`](#gen.Chunk.start_patch)
    - [`gen.Chunk.then`](#gen.Chunk.then)
    - [`gen.Chunk.time_reversed`](#gen.Chunk.time_reversed)
    - [`gen.Chunk.to_closed_circuit`](#gen.Chunk.to_closed_circuit)
    - [`gen.Chunk.to_html_viewer`](#gen.Chunk.to_html_viewer)
    - [`gen.Chunk.verify`](#gen.Chunk.verify)
    - [`gen.Chunk.verify_distance_is_at_least_2`](#gen.Chunk.verify_distance_is_at_least_2)
    - [`gen.Chunk.verify_distance_is_at_least_3`](#gen.Chunk.verify_distance_is_at_least_3)
    - [`gen.Chunk.with_edits`](#gen.Chunk.with_edits)
    - [`gen.Chunk.with_flag_added_to_all_flows`](#gen.Chunk.with_flag_added_to_all_flows)
    - [`gen.Chunk.with_obs_flows_as_det_flows`](#gen.Chunk.with_obs_flows_as_det_flows)
    - [`gen.Chunk.with_repetitions`](#gen.Chunk.with_repetitions)
    - [`gen.Chunk.with_transformed_coords`](#gen.Chunk.with_transformed_coords)
    - [`gen.Chunk.with_xz_flipped`](#gen.Chunk.with_xz_flipped)
- [`gen.ChunkBuilder`](#gen.ChunkBuilder)
    - [`gen.ChunkBuilder.__init__`](#gen.ChunkBuilder.__init__)
    - [`gen.ChunkBuilder.add_discarded_flow_input`](#gen.ChunkBuilder.add_discarded_flow_input)
    - [`gen.ChunkBuilder.add_discarded_flow_output`](#gen.ChunkBuilder.add_discarded_flow_output)
    - [`gen.ChunkBuilder.add_flow`](#gen.ChunkBuilder.add_flow)
    - [`gen.ChunkBuilder.append`](#gen.ChunkBuilder.append)
    - [`gen.ChunkBuilder.classical_paulis`](#gen.ChunkBuilder.classical_paulis)
    - [`gen.ChunkBuilder.demolition_measure_with_feedback_passthrough`](#gen.ChunkBuilder.demolition_measure_with_feedback_passthrough)
    - [`gen.ChunkBuilder.finish_chunk`](#gen.ChunkBuilder.finish_chunk)
    - [`gen.ChunkBuilder.finish_chunk_keep_semi_auto`](#gen.ChunkBuilder.finish_chunk_keep_semi_auto)
    - [`gen.ChunkBuilder.lookup_mids`](#gen.ChunkBuilder.lookup_mids)
    - [`gen.ChunkBuilder.record_measurement_group`](#gen.ChunkBuilder.record_measurement_group)
- [`gen.ChunkCompiler`](#gen.ChunkCompiler)
    - [`gen.ChunkCompiler.__init__`](#gen.ChunkCompiler.__init__)
    - [`gen.ChunkCompiler.append`](#gen.ChunkCompiler.append)
    - [`gen.ChunkCompiler.append_magic_end_chunk`](#gen.ChunkCompiler.append_magic_end_chunk)
    - [`gen.ChunkCompiler.append_magic_init_chunk`](#gen.ChunkCompiler.append_magic_init_chunk)
    - [`gen.ChunkCompiler.copy`](#gen.ChunkCompiler.copy)
    - [`gen.ChunkCompiler.cur_circuit_html_viewer`](#gen.ChunkCompiler.cur_circuit_html_viewer)
    - [`gen.ChunkCompiler.cur_end_interface`](#gen.ChunkCompiler.cur_end_interface)
    - [`gen.ChunkCompiler.ensure_qubits_included`](#gen.ChunkCompiler.ensure_qubits_included)
    - [`gen.ChunkCompiler.finish_circuit`](#gen.ChunkCompiler.finish_circuit)
- [`gen.ChunkInterface`](#gen.ChunkInterface)
    - [`gen.ChunkInterface.data_set`](#gen.ChunkInterface.data_set)
    - [`gen.ChunkInterface.partitioned_detector_flows`](#gen.ChunkInterface.partitioned_detector_flows)
    - [`gen.ChunkInterface.to_code`](#gen.ChunkInterface.to_code)
    - [`gen.ChunkInterface.to_patch`](#gen.ChunkInterface.to_patch)
    - [`gen.ChunkInterface.to_svg`](#gen.ChunkInterface.to_svg)
    - [`gen.ChunkInterface.used_set`](#gen.ChunkInterface.used_set)
    - [`gen.ChunkInterface.with_discards_as_ports`](#gen.ChunkInterface.with_discards_as_ports)
    - [`gen.ChunkInterface.with_edits`](#gen.ChunkInterface.with_edits)
    - [`gen.ChunkInterface.with_transformed_coords`](#gen.ChunkInterface.with_transformed_coords)
    - [`gen.ChunkInterface.without_discards`](#gen.ChunkInterface.without_discards)
    - [`gen.ChunkInterface.without_keyed`](#gen.ChunkInterface.without_keyed)
- [`gen.ChunkLoop`](#gen.ChunkLoop)
    - [`gen.ChunkLoop.end_interface`](#gen.ChunkLoop.end_interface)
    - [`gen.ChunkLoop.end_patch`](#gen.ChunkLoop.end_patch)
    - [`gen.ChunkLoop.find_distance`](#gen.ChunkLoop.find_distance)
    - [`gen.ChunkLoop.find_logical_error`](#gen.ChunkLoop.find_logical_error)
    - [`gen.ChunkLoop.flattened`](#gen.ChunkLoop.flattened)
    - [`gen.ChunkLoop.start_interface`](#gen.ChunkLoop.start_interface)
    - [`gen.ChunkLoop.start_patch`](#gen.ChunkLoop.start_patch)
    - [`gen.ChunkLoop.time_reversed`](#gen.ChunkLoop.time_reversed)
    - [`gen.ChunkLoop.to_closed_circuit`](#gen.ChunkLoop.to_closed_circuit)
    - [`gen.ChunkLoop.to_html_viewer`](#gen.ChunkLoop.to_html_viewer)
    - [`gen.ChunkLoop.verify`](#gen.ChunkLoop.verify)
    - [`gen.ChunkLoop.verify_distance_is_at_least_2`](#gen.ChunkLoop.verify_distance_is_at_least_2)
    - [`gen.ChunkLoop.verify_distance_is_at_least_3`](#gen.ChunkLoop.verify_distance_is_at_least_3)
    - [`gen.ChunkLoop.with_repetitions`](#gen.ChunkLoop.with_repetitions)
- [`gen.ChunkReflow`](#gen.ChunkReflow)
    - [`gen.ChunkReflow.end_code`](#gen.ChunkReflow.end_code)
    - [`gen.ChunkReflow.end_interface`](#gen.ChunkReflow.end_interface)
    - [`gen.ChunkReflow.end_patch`](#gen.ChunkReflow.end_patch)
    - [`gen.ChunkReflow.flattened`](#gen.ChunkReflow.flattened)
    - [`gen.ChunkReflow.from_auto_rewrite`](#gen.ChunkReflow.from_auto_rewrite)
    - [`gen.ChunkReflow.from_auto_rewrite_transitions_using_stable`](#gen.ChunkReflow.from_auto_rewrite_transitions_using_stable)
    - [`gen.ChunkReflow.removed_inputs`](#gen.ChunkReflow.removed_inputs)
    - [`gen.ChunkReflow.start_code`](#gen.ChunkReflow.start_code)
    - [`gen.ChunkReflow.start_interface`](#gen.ChunkReflow.start_interface)
    - [`gen.ChunkReflow.start_patch`](#gen.ChunkReflow.start_patch)
    - [`gen.ChunkReflow.verify`](#gen.ChunkReflow.verify)
    - [`gen.ChunkReflow.with_obs_flows_as_det_flows`](#gen.ChunkReflow.with_obs_flows_as_det_flows)
    - [`gen.ChunkReflow.with_transformed_coords`](#gen.ChunkReflow.with_transformed_coords)
- [`gen.ChunkSemiAuto`](#gen.ChunkSemiAuto)
    - [`gen.ChunkSemiAuto.__init__`](#gen.ChunkSemiAuto.__init__)
    - [`gen.ChunkSemiAuto.solve`](#gen.ChunkSemiAuto.solve)
- [`gen.ColoredLineData`](#gen.ColoredLineData)
    - [`gen.ColoredLineData.fused`](#gen.ColoredLineData.fused)
- [`gen.ColoredTriangleData`](#gen.ColoredTriangleData)
    - [`gen.ColoredTriangleData.fused`](#gen.ColoredTriangleData.fused)
    - [`gen.ColoredTriangleData.square`](#gen.ColoredTriangleData.square)
- [`gen.Flow`](#gen.Flow)
    - [`gen.Flow.__init__`](#gen.Flow.__init__)
    - [`gen.Flow.fuse_with_next_flow`](#gen.Flow.fuse_with_next_flow)
    - [`gen.Flow.key_end`](#gen.Flow.key_end)
    - [`gen.Flow.key_start`](#gen.Flow.key_start)
    - [`gen.Flow.to_stim_flow`](#gen.Flow.to_stim_flow)
    - [`gen.Flow.with_edits`](#gen.Flow.with_edits)
    - [`gen.Flow.with_transformed_coords`](#gen.Flow.with_transformed_coords)
    - [`gen.Flow.with_xz_flipped`](#gen.Flow.with_xz_flipped)
- [`gen.FlowMetadata`](#gen.FlowMetadata)
    - [`gen.FlowMetadata.__init__`](#gen.FlowMetadata.__init__)
- [`gen.FlowSemiAuto`](#gen.FlowSemiAuto)
    - [`gen.FlowSemiAuto.__init__`](#gen.FlowSemiAuto.__init__)
    - [`gen.FlowSemiAuto.to_flow`](#gen.FlowSemiAuto.to_flow)
    - [`gen.FlowSemiAuto.to_stim_flow`](#gen.FlowSemiAuto.to_stim_flow)
    - [`gen.FlowSemiAuto.with_edits`](#gen.FlowSemiAuto.with_edits)
- [`gen.InteractLayer`](#gen.InteractLayer)
    - [`gen.InteractLayer.append_into_stim_circuit`](#gen.InteractLayer.append_into_stim_circuit)
    - [`gen.InteractLayer.copy`](#gen.InteractLayer.copy)
    - [`gen.InteractLayer.locally_optimized`](#gen.InteractLayer.locally_optimized)
    - [`gen.InteractLayer.rotate_to_z_layer`](#gen.InteractLayer.rotate_to_z_layer)
    - [`gen.InteractLayer.to_z_basis`](#gen.InteractLayer.to_z_basis)
    - [`gen.InteractLayer.touched`](#gen.InteractLayer.touched)
- [`gen.LayerCircuit`](#gen.LayerCircuit)
    - [`gen.LayerCircuit.copy`](#gen.LayerCircuit.copy)
    - [`gen.LayerCircuit.from_stim_circuit`](#gen.LayerCircuit.from_stim_circuit)
    - [`gen.LayerCircuit.to_stim_circuit`](#gen.LayerCircuit.to_stim_circuit)
    - [`gen.LayerCircuit.to_z_basis`](#gen.LayerCircuit.to_z_basis)
    - [`gen.LayerCircuit.touched`](#gen.LayerCircuit.touched)
    - [`gen.LayerCircuit.with_cleaned_up_loop_iterations`](#gen.LayerCircuit.with_cleaned_up_loop_iterations)
    - [`gen.LayerCircuit.with_clearable_rotation_layers_cleared`](#gen.LayerCircuit.with_clearable_rotation_layers_cleared)
    - [`gen.LayerCircuit.with_ejected_loop_iterations`](#gen.LayerCircuit.with_ejected_loop_iterations)
    - [`gen.LayerCircuit.with_irrelevant_tail_layers_removed`](#gen.LayerCircuit.with_irrelevant_tail_layers_removed)
    - [`gen.LayerCircuit.with_locally_merged_measure_layers`](#gen.LayerCircuit.with_locally_merged_measure_layers)
    - [`gen.LayerCircuit.with_locally_optimized_layers`](#gen.LayerCircuit.with_locally_optimized_layers)
    - [`gen.LayerCircuit.with_qubit_coords_at_start`](#gen.LayerCircuit.with_qubit_coords_at_start)
    - [`gen.LayerCircuit.with_rotations_before_resets_removed`](#gen.LayerCircuit.with_rotations_before_resets_removed)
    - [`gen.LayerCircuit.with_rotations_merged_earlier`](#gen.LayerCircuit.with_rotations_merged_earlier)
    - [`gen.LayerCircuit.with_rotations_rolled_from_end_of_loop_to_start_of_loop`](#gen.LayerCircuit.with_rotations_rolled_from_end_of_loop_to_start_of_loop)
    - [`gen.LayerCircuit.with_whole_layers_slid_as_early_as_possible_for_merge_with_same_layer`](#gen.LayerCircuit.with_whole_layers_slid_as_early_as_possible_for_merge_with_same_layer)
    - [`gen.LayerCircuit.with_whole_layers_slid_as_to_merge_with_previous_layer_of_same_type`](#gen.LayerCircuit.with_whole_layers_slid_as_to_merge_with_previous_layer_of_same_type)
    - [`gen.LayerCircuit.with_whole_rotation_layers_slid_earlier`](#gen.LayerCircuit.with_whole_rotation_layers_slid_earlier)
    - [`gen.LayerCircuit.without_empty_layers`](#gen.LayerCircuit.without_empty_layers)
- [`gen.MeasureLayer`](#gen.MeasureLayer)
    - [`gen.MeasureLayer.append_into_stim_circuit`](#gen.MeasureLayer.append_into_stim_circuit)
    - [`gen.MeasureLayer.copy`](#gen.MeasureLayer.copy)
    - [`gen.MeasureLayer.locally_optimized`](#gen.MeasureLayer.locally_optimized)
    - [`gen.MeasureLayer.to_z_basis`](#gen.MeasureLayer.to_z_basis)
    - [`gen.MeasureLayer.touched`](#gen.MeasureLayer.touched)
- [`gen.NoiseModel`](#gen.NoiseModel)
    - [`gen.NoiseModel.noisy_circuit`](#gen.NoiseModel.noisy_circuit)
    - [`gen.NoiseModel.noisy_circuit_skipping_mpp_boundaries`](#gen.NoiseModel.noisy_circuit_skipping_mpp_boundaries)
    - [`gen.NoiseModel.si1000`](#gen.NoiseModel.si1000)
    - [`gen.NoiseModel.uniform_depolarizing`](#gen.NoiseModel.uniform_depolarizing)
- [`gen.NoiseRule`](#gen.NoiseRule)
    - [`gen.NoiseRule.__init__`](#gen.NoiseRule.__init__)
    - [`gen.NoiseRule.append_noisy_version_of`](#gen.NoiseRule.append_noisy_version_of)
- [`gen.Patch`](#gen.Patch)
    - [`gen.Patch.data_set`](#gen.Patch.data_set)
    - [`gen.Patch.m2tile`](#gen.Patch.m2tile)
    - [`gen.Patch.measure_set`](#gen.Patch.measure_set)
    - [`gen.Patch.partitioned_tiles`](#gen.Patch.partitioned_tiles)
    - [`gen.Patch.to_svg`](#gen.Patch.to_svg)
    - [`gen.Patch.used_set`](#gen.Patch.used_set)
    - [`gen.Patch.with_edits`](#gen.Patch.with_edits)
    - [`gen.Patch.with_only_x_tiles`](#gen.Patch.with_only_x_tiles)
    - [`gen.Patch.with_only_y_tiles`](#gen.Patch.with_only_y_tiles)
    - [`gen.Patch.with_only_z_tiles`](#gen.Patch.with_only_z_tiles)
    - [`gen.Patch.with_remaining_degrees_of_freedom_as_logicals`](#gen.Patch.with_remaining_degrees_of_freedom_as_logicals)
    - [`gen.Patch.with_transformed_bases`](#gen.Patch.with_transformed_bases)
    - [`gen.Patch.with_transformed_coords`](#gen.Patch.with_transformed_coords)
    - [`gen.Patch.with_xz_flipped`](#gen.Patch.with_xz_flipped)
- [`gen.PauliMap`](#gen.PauliMap)
    - [`gen.PauliMap.__init__`](#gen.PauliMap.__init__)
    - [`gen.PauliMap.anticommutes`](#gen.PauliMap.anticommutes)
    - [`gen.PauliMap.commutes`](#gen.PauliMap.commutes)
    - [`gen.PauliMap.from_xs`](#gen.PauliMap.from_xs)
    - [`gen.PauliMap.from_ys`](#gen.PauliMap.from_ys)
    - [`gen.PauliMap.from_zs`](#gen.PauliMap.from_zs)
    - [`gen.PauliMap.get`](#gen.PauliMap.get)
    - [`gen.PauliMap.items`](#gen.PauliMap.items)
    - [`gen.PauliMap.keys`](#gen.PauliMap.keys)
    - [`gen.PauliMap.to_stim_pauli_string`](#gen.PauliMap.to_stim_pauli_string)
    - [`gen.PauliMap.to_stim_targets`](#gen.PauliMap.to_stim_targets)
    - [`gen.PauliMap.to_tile`](#gen.PauliMap.to_tile)
    - [`gen.PauliMap.values`](#gen.PauliMap.values)
    - [`gen.PauliMap.with_basis`](#gen.PauliMap.with_basis)
    - [`gen.PauliMap.with_name`](#gen.PauliMap.with_name)
    - [`gen.PauliMap.with_transformed_coords`](#gen.PauliMap.with_transformed_coords)
    - [`gen.PauliMap.with_xy_flipped`](#gen.PauliMap.with_xy_flipped)
    - [`gen.PauliMap.with_xz_flipped`](#gen.PauliMap.with_xz_flipped)
- [`gen.ResetLayer`](#gen.ResetLayer)
    - [`gen.ResetLayer.append_into_stim_circuit`](#gen.ResetLayer.append_into_stim_circuit)
    - [`gen.ResetLayer.copy`](#gen.ResetLayer.copy)
    - [`gen.ResetLayer.locally_optimized`](#gen.ResetLayer.locally_optimized)
    - [`gen.ResetLayer.to_z_basis`](#gen.ResetLayer.to_z_basis)
    - [`gen.ResetLayer.touched`](#gen.ResetLayer.touched)
- [`gen.RotationLayer`](#gen.RotationLayer)
    - [`gen.RotationLayer.append_into_stim_circuit`](#gen.RotationLayer.append_into_stim_circuit)
    - [`gen.RotationLayer.append_named_rotation`](#gen.RotationLayer.append_named_rotation)
    - [`gen.RotationLayer.copy`](#gen.RotationLayer.copy)
    - [`gen.RotationLayer.inverse`](#gen.RotationLayer.inverse)
    - [`gen.RotationLayer.is_vacuous`](#gen.RotationLayer.is_vacuous)
    - [`gen.RotationLayer.locally_optimized`](#gen.RotationLayer.locally_optimized)
    - [`gen.RotationLayer.prepend_named_rotation`](#gen.RotationLayer.prepend_named_rotation)
    - [`gen.RotationLayer.touched`](#gen.RotationLayer.touched)
- [`gen.StabilizerCode`](#gen.StabilizerCode)
    - [`gen.StabilizerCode.__init__`](#gen.StabilizerCode.__init__)
    - [`gen.StabilizerCode.as_interface`](#gen.StabilizerCode.as_interface)
    - [`gen.StabilizerCode.concat_over`](#gen.StabilizerCode.concat_over)
    - [`gen.StabilizerCode.data_set`](#gen.StabilizerCode.data_set)
    - [`gen.StabilizerCode.find_distance`](#gen.StabilizerCode.find_distance)
    - [`gen.StabilizerCode.find_logical_error`](#gen.StabilizerCode.find_logical_error)
    - [`gen.StabilizerCode.flat_logicals`](#gen.StabilizerCode.flat_logicals)
    - [`gen.StabilizerCode.from_patch_with_inferred_observables`](#gen.StabilizerCode.from_patch_with_inferred_observables)
    - [`gen.StabilizerCode.get_observable_by_basis`](#gen.StabilizerCode.get_observable_by_basis)
    - [`gen.StabilizerCode.list_pure_basis_observables`](#gen.StabilizerCode.list_pure_basis_observables)
    - [`gen.StabilizerCode.make_code_capacity_circuit`](#gen.StabilizerCode.make_code_capacity_circuit)
    - [`gen.StabilizerCode.make_phenom_circuit`](#gen.StabilizerCode.make_phenom_circuit)
    - [`gen.StabilizerCode.measure_set`](#gen.StabilizerCode.measure_set)
    - [`gen.StabilizerCode.patch`](#gen.StabilizerCode.patch)
    - [`gen.StabilizerCode.physical_to_logical`](#gen.StabilizerCode.physical_to_logical)
    - [`gen.StabilizerCode.tiles`](#gen.StabilizerCode.tiles)
    - [`gen.StabilizerCode.to_svg`](#gen.StabilizerCode.to_svg)
    - [`gen.StabilizerCode.transversal_init_chunk`](#gen.StabilizerCode.transversal_init_chunk)
    - [`gen.StabilizerCode.transversal_measure_chunk`](#gen.StabilizerCode.transversal_measure_chunk)
    - [`gen.StabilizerCode.used_set`](#gen.StabilizerCode.used_set)
    - [`gen.StabilizerCode.verify`](#gen.StabilizerCode.verify)
    - [`gen.StabilizerCode.verify_distance_is_at_least_2`](#gen.StabilizerCode.verify_distance_is_at_least_2)
    - [`gen.StabilizerCode.verify_distance_is_at_least_3`](#gen.StabilizerCode.verify_distance_is_at_least_3)
    - [`gen.StabilizerCode.with_edits`](#gen.StabilizerCode.with_edits)
    - [`gen.StabilizerCode.with_integer_coordinates`](#gen.StabilizerCode.with_integer_coordinates)
    - [`gen.StabilizerCode.with_observables_from_basis`](#gen.StabilizerCode.with_observables_from_basis)
    - [`gen.StabilizerCode.with_remaining_degrees_of_freedom_as_logicals`](#gen.StabilizerCode.with_remaining_degrees_of_freedom_as_logicals)
    - [`gen.StabilizerCode.with_transformed_coords`](#gen.StabilizerCode.with_transformed_coords)
    - [`gen.StabilizerCode.with_xz_flipped`](#gen.StabilizerCode.with_xz_flipped)
    - [`gen.StabilizerCode.x_basis_subset`](#gen.StabilizerCode.x_basis_subset)
    - [`gen.StabilizerCode.z_basis_subset`](#gen.StabilizerCode.z_basis_subset)
- [`gen.StimCircuitLoom`](#gen.StimCircuitLoom)
    - [`gen.StimCircuitLoom.weave`](#gen.StimCircuitLoom.weave)
    - [`gen.StimCircuitLoom.weaved_target_rec_from_c0`](#gen.StimCircuitLoom.weaved_target_rec_from_c0)
    - [`gen.StimCircuitLoom.weaved_target_rec_from_c1`](#gen.StimCircuitLoom.weaved_target_rec_from_c1)
- [`gen.TextData`](#gen.TextData)
- [`gen.Tile`](#gen.Tile)
    - [`gen.Tile.__init__`](#gen.Tile.__init__)
    - [`gen.Tile.basis`](#gen.Tile.basis)
    - [`gen.Tile.center`](#gen.Tile.center)
    - [`gen.Tile.data_set`](#gen.Tile.data_set)
    - [`gen.Tile.to_pauli_map`](#gen.Tile.to_pauli_map)
    - [`gen.Tile.used_set`](#gen.Tile.used_set)
    - [`gen.Tile.with_bases`](#gen.Tile.with_bases)
    - [`gen.Tile.with_basis`](#gen.Tile.with_basis)
    - [`gen.Tile.with_data_qubit_cleared`](#gen.Tile.with_data_qubit_cleared)
    - [`gen.Tile.with_edits`](#gen.Tile.with_edits)
    - [`gen.Tile.with_transformed_bases`](#gen.Tile.with_transformed_bases)
    - [`gen.Tile.with_transformed_coords`](#gen.Tile.with_transformed_coords)
    - [`gen.Tile.with_xz_flipped`](#gen.Tile.with_xz_flipped)
- [`gen.append_reindexed_content_to_circuit`](#gen.append_reindexed_content_to_circuit)
- [`gen.circuit_to_cycle_code_slices`](#gen.circuit_to_cycle_code_slices)
- [`gen.circuit_to_dem_target_measurement_records_map`](#gen.circuit_to_dem_target_measurement_records_map)
- [`gen.circuit_with_xz_flipped`](#gen.circuit_with_xz_flipped)
- [`gen.compile_chunks_into_circuit`](#gen.compile_chunks_into_circuit)
- [`gen.complex_key`](#gen.complex_key)
- [`gen.count_measurement_layers`](#gen.count_measurement_layers)
- [`gen.find_d1_error`](#gen.find_d1_error)
- [`gen.find_d2_error`](#gen.find_d2_error)
- [`gen.gate_counts_for_circuit`](#gen.gate_counts_for_circuit)
- [`gen.gates_used_by_circuit`](#gen.gates_used_by_circuit)
- [`gen.gltf_model`](#gen.gltf_model)
    - [`gen.gltf_model.write_viewer_to`](#gen.gltf_model.write_viewer_to)
- [`gen.gltf_model_from_colored_triangle_data`](#gen.gltf_model_from_colored_triangle_data)
- [`gen.min_max_complex`](#gen.min_max_complex)
- [`gen.sorted_complex`](#gen.sorted_complex)
- [`gen.stim_circuit_html_viewer`](#gen.stim_circuit_html_viewer)
- [`gen.stim_circuit_with_transformed_coords`](#gen.stim_circuit_with_transformed_coords)
- [`gen.stim_circuit_with_transformed_moments`](#gen.stim_circuit_with_transformed_moments)
- [`gen.str_html`](#gen.str_html)
    - [`gen.str_html.write_to`](#gen.str_html.write_to)
- [`gen.str_svg`](#gen.str_svg)
    - [`gen.str_svg.write_to`](#gen.str_svg.write_to)
- [`gen.svg`](#gen.svg)
- [`gen.transpile_to_z_basis_interaction_circuit`](#gen.transpile_to_z_basis_interaction_circuit)
- [`gen.transversal_code_transition_chunks`](#gen.transversal_code_transition_chunks)
- [`gen.verify_distance_is_at_least_2`](#gen.verify_distance_is_at_least_2)
- [`gen.verify_distance_is_at_least_3`](#gen.verify_distance_is_at_least_3)
- [`gen.viz_3d_gltf_model_html`](#gen.viz_3d_gltf_model_html)
- [`gen.xor_sorted`](#gen.xor_sorted)
```python
# Types used by the method definitions.
from typing import overload, TYPE_CHECKING, Any, Iterable
import io
import pathlib
import numpy as np
```

<a name="gen.Chunk"></a>
```python
# gen.Chunk

# (at top-level in the gen module)
class Chunk:
    """A circuit chunk with accompanying stabilizer flow assertions.
    """
```

<a name="gen.Chunk.__init__"></a>
```python
# gen.Chunk.__init__

# (in class gen.Chunk)
def __init__(
    self,
    circuit: 'stim.Circuit',
    *,
    flows: 'Iterable[Flow]',
    discarded_inputs: 'Iterable[PauliMap | Tile]' = (),
    discarded_outputs: 'Iterable[PauliMap | Tile]' = (),
    wants_to_merge_with_next: 'bool' = False,
    wants_to_merge_with_prev: 'bool' = False,
    q2i: 'dict[complex, int] | None' = None,
    o2i: 'dict[Any, int] | None' = None,
):
    """

    Args:
        circuit: The circuit implementing the chunk's functionality.
        flows: A series of stabilizer flows that the circuit implements.
        discarded_inputs: Explicitly rejected in flows. For example, a data
            measurement chunk might reject flows for stabilizers from the
            anticommuting basis. If they are not rejected, then compilation
            will fail when attempting to combine this chunk with a preceding
            chunk that has those stabilizers from the anticommuting basis
            flowing out.
        discarded_outputs: Explicitly rejected out flows. For example, an
            initialization chunk might reject flows for stabilizers from the
            anticommuting basis. If they are not rejected, then compilation
            will fail when attempting to combine this chunk with a following
            chunk that has those stabilizers from the anticommuting basis
            flowing in.
        wants_to_merge_with_next: Defaults to False. When set to True,
            the chunk compiler won't insert a TICK between this chunk
            and the next chunk.
        wants_to_merge_with_prev: Defaults to False. When set to True,
            the chunk compiler won't insert a TICK between this chunk
            and the previous chunk.
        q2i: Defaults to None (infer from QUBIT_COORDS instructions in circuit else
            raise an exception). The gen-qubit-coordinate-to-stim-qubit-index mapping
            used to translate between gen's qubit keys and stim's qubit keys.
        o2i: Defaults to None (raise an exception if observables present in circuit).
            The gen-observable-key-to-stim-observable-index mapping used to translate
            between gen's observable keys and stim's observable keys.
    """
```

<a name="gen.Chunk.end_code"></a>
```python
# gen.Chunk.end_code

# (in class gen.Chunk)
def end_code(
    self,
) -> 'StabilizerCode':
```

<a name="gen.Chunk.end_interface"></a>
```python
# gen.Chunk.end_interface

# (in class gen.Chunk)
def end_interface(
    self,
    *,
    skip_passthroughs: 'bool' = False,
) -> 'ChunkInterface':
    """Returns a description of the flows that should exit from the chunk.
    """
```

<a name="gen.Chunk.end_patch"></a>
```python
# gen.Chunk.end_patch

# (in class gen.Chunk)
def end_patch(
    self,
) -> 'Patch':
```

<a name="gen.Chunk.find_distance"></a>
```python
# gen.Chunk.find_distance

# (in class gen.Chunk)
def find_distance(
    self,
    *,
    max_search_weight: 'int',
    noise: 'float | NoiseModel' = 0.001,
    noiseless_qubits: 'Iterable[float | int | complex]' = (),
    skip_adding_noise: 'bool' = False,
) -> 'int':
```

<a name="gen.Chunk.find_logical_error"></a>
```python
# gen.Chunk.find_logical_error

# (in class gen.Chunk)
def find_logical_error(
    self,
    *,
    max_search_weight: 'int',
    noise: 'float | NoiseModel' = 0.001,
    noiseless_qubits: 'Iterable[float | int | complex]' = (),
    skip_adding_noise: 'bool' = False,
) -> 'list[stim.ExplainedError]':
```

<a name="gen.Chunk.flattened"></a>
```python
# gen.Chunk.flattened

# (in class gen.Chunk)
def flattened(
    self,
) -> 'list[Chunk]':
    """This is here for duck-type compatibility with ChunkLoop.
    """
```

<a name="gen.Chunk.from_circuit_with_mpp_boundaries"></a>
```python
# gen.Chunk.from_circuit_with_mpp_boundaries

# (in class gen.Chunk)
def from_circuit_with_mpp_boundaries(
    circuit: 'stim.Circuit',
) -> 'Chunk':
```

<a name="gen.Chunk.start_code"></a>
```python
# gen.Chunk.start_code

# (in class gen.Chunk)
def start_code(
    self,
) -> 'StabilizerCode':
```

<a name="gen.Chunk.start_interface"></a>
```python
# gen.Chunk.start_interface

# (in class gen.Chunk)
def start_interface(
    self,
    *,
    skip_passthroughs: 'bool' = False,
) -> 'ChunkInterface':
    """Returns a description of the flows that should enter into the chunk.
    """
```

<a name="gen.Chunk.start_patch"></a>
```python
# gen.Chunk.start_patch

# (in class gen.Chunk)
def start_patch(
    self,
) -> 'Patch':
```

<a name="gen.Chunk.then"></a>
```python
# gen.Chunk.then

# (in class gen.Chunk)
def then(
    self,
    other: 'Chunk | ChunkReflow | ChunkLoop',
) -> 'Chunk':
```

<a name="gen.Chunk.time_reversed"></a>
```python
# gen.Chunk.time_reversed

# (in class gen.Chunk)
def time_reversed(
    self,
) -> 'Chunk':
    """Checks that this chunk's circuit actually implements its flows.
    """
```

<a name="gen.Chunk.to_closed_circuit"></a>
```python
# gen.Chunk.to_closed_circuit

# (in class gen.Chunk)
def to_closed_circuit(
    self,
) -> 'stim.Circuit':
    """Compiles the chunk into a circuit by conjugating with mpp init/end chunks.
    """
```

<a name="gen.Chunk.to_html_viewer"></a>
```python
# gen.Chunk.to_html_viewer

# (in class gen.Chunk)
def to_html_viewer(
    self,
    *,
    patch: 'Patch | StabilizerCode | ChunkInterface | dict[int, Patch | StabilizerCode | ChunkInterface] | None' = None,
    tile_color_func: 'Callable[[Tile], tuple[float, float, float, float]] | None' = None,
    known_error: 'Iterable[stim.ExplainedError] | None' = None,
) -> 'str_html':
```

<a name="gen.Chunk.verify"></a>
```python
# gen.Chunk.verify

# (in class gen.Chunk)
def verify(
    self,
    *,
    expected_in: 'ChunkInterface | StabilizerCode | Patch | None' = None,
    expected_out: 'ChunkInterface | StabilizerCode | Patch | None' = None,
    should_measure_all_code_stabilizers: 'bool' = False,
    allow_overlapping_flows: 'bool' = False,
):
    """Checks that this chunk's circuit actually implements its flows.
    """
```

<a name="gen.Chunk.verify_distance_is_at_least_2"></a>
```python
# gen.Chunk.verify_distance_is_at_least_2

# (in class gen.Chunk)
def verify_distance_is_at_least_2(
    self,
    *,
    noise: 'float | NoiseModel' = 0.001,
):
    """Verifies undetected logical errors require at least 2 physical errors.

    By default, verifies using a uniform depolarizing circuit noise model.
    """
```

<a name="gen.Chunk.verify_distance_is_at_least_3"></a>
```python
# gen.Chunk.verify_distance_is_at_least_3

# (in class gen.Chunk)
def verify_distance_is_at_least_3(
    self,
    *,
    noise: 'float | NoiseModel' = 0.001,
):
    """Verifies undetected logical errors require at least 3 physical errors.

    By default, verifies using a uniform depolarizing circuit noise model.
    """
```

<a name="gen.Chunk.with_edits"></a>
```python
# gen.Chunk.with_edits

# (in class gen.Chunk)
def with_edits(
    self,
    *,
    circuit: 'stim.Circuit | None' = None,
    q2i: 'dict[complex, int] | None' = None,
    flows: 'Iterable[Flow] | None' = None,
    discarded_inputs: 'Iterable[PauliMap] | None' = None,
    discarded_outputs: 'Iterable[PauliMap] | None' = None,
    wants_to_merge_with_prev: 'bool | None' = None,
    wants_to_merge_with_next: 'bool | None' = None,
) -> 'Chunk':
```

<a name="gen.Chunk.with_flag_added_to_all_flows"></a>
```python
# gen.Chunk.with_flag_added_to_all_flows

# (in class gen.Chunk)
def with_flag_added_to_all_flows(
    self,
    flag: 'str',
) -> 'Chunk':
```

<a name="gen.Chunk.with_obs_flows_as_det_flows"></a>
```python
# gen.Chunk.with_obs_flows_as_det_flows

# (in class gen.Chunk)
def with_obs_flows_as_det_flows(
    self,
) -> 'Chunk':
```

<a name="gen.Chunk.with_repetitions"></a>
```python
# gen.Chunk.with_repetitions

# (in class gen.Chunk)
def with_repetitions(
    self,
    repetitions: 'int',
) -> 'ChunkLoop':
```

<a name="gen.Chunk.with_transformed_coords"></a>
```python
# gen.Chunk.with_transformed_coords

# (in class gen.Chunk)
def with_transformed_coords(
    self,
    transform: 'Callable[[complex], complex]',
) -> 'Chunk':
```

<a name="gen.Chunk.with_xz_flipped"></a>
```python
# gen.Chunk.with_xz_flipped

# (in class gen.Chunk)
def with_xz_flipped(
    self,
) -> 'Chunk':
```

<a name="gen.ChunkBuilder"></a>
```python
# gen.ChunkBuilder

# (at top-level in the gen module)
class ChunkBuilder:
    """Helper class for building stim circuits.

    Handles qubit indexing (complex -> int).
    Handles measurement tracking (naming results and referring to them by name).
    """
```

<a name="gen.ChunkBuilder.__init__"></a>
```python
# gen.ChunkBuilder.__init__

# (in class gen.ChunkBuilder)
def __init__(
    self,
    allowed_qubits: 'Iterable[complex] | None' = None,
    *,
    unknown_qubit_append_mode: "Literal['auto', 'error', 'skip', 'include']" = 'auto',
):
    """Creates a Builder for creating a circuit over the given qubits.

    Args:
        allowed_qubits: Defaults to None (everything allowed). Specifies the qubit positions
            that the circuit is permitted to contain.
        unknown_qubit_append_mode: Defaults to 'auto'. The available options are:
            - 'error': When a qubit position outside `allowed_qubits` is encountered,
                raise an exception.
            - 'include': When a qubit position outside `allowed_qubits` is encountered,
                automatically add it into `builder.q2i` and `builder.allowed_qubits`.
            - 'skip': When a qubit position outside `allowed_qubits` is encountered,
                ignore it. Note that, for two-qubit and multi-qubit operations, this
                will ignore the pair or group of targets containing the skipped position.
            - 'auto':  Replace by 'include' if allowed_qubits is None else 'error'.
    """
```

<a name="gen.ChunkBuilder.add_discarded_flow_input"></a>
```python
# gen.ChunkBuilder.add_discarded_flow_input

# (in class gen.ChunkBuilder)
def add_discarded_flow_input(
    self,
    flow: 'PauliMap | Tile',
) -> 'None':
```

<a name="gen.ChunkBuilder.add_discarded_flow_output"></a>
```python
# gen.ChunkBuilder.add_discarded_flow_output

# (in class gen.ChunkBuilder)
def add_discarded_flow_output(
    self,
    flow: 'PauliMap | Tile',
) -> 'None':
```

<a name="gen.ChunkBuilder.add_flow"></a>
```python
# gen.ChunkBuilder.add_flow

# (in class gen.ChunkBuilder)
def add_flow(
    self,
    *,
    start: "PauliMap | Tile | Literal['auto'] | None" = None,
    end: "PauliMap | Tile | Literal['auto'] | None" = None,
    ms: "Iterable[Any] | Literal['auto']" = (),
    ignore_unmatched_ms: 'bool' = False,
    obs_key: 'Any' = None,
    center: 'complex | None' = None,
    flags: 'Iterable[str]' = frozenset(),
    sign: 'bool | None' = None,
) -> 'None':
```

<a name="gen.ChunkBuilder.append"></a>
```python
# gen.ChunkBuilder.append

# (in class gen.ChunkBuilder)
def append(
    self,
    gate: 'str',
    targets: 'Iterable[complex | Sequence[complex] | PauliMap | Tile | Any]' = (),
    *,
    arg: 'float | Iterable[float] | None' = None,
    measure_key_func: 'Callable[[complex], Any] | Callable[[tuple[complex, complex]], Any] | Callable[[PauliMap | Tile], Any] | None' = lambda e: e,
    tag: 'str' = '',
) -> 'None':
    """Appends an instruction to the builder's circuit.

    This method differs from `stim.Circuit.append` in the following ways:

    1) It targets qubits by position instead of by index. Also, it takes two
    qubit targets as pairs instead of interleaved. For example, instead of
    saying

        a = builder.q2i[5 + 1j]
        b = builder.q2i[5]
        c = builder.q2i[0]
        d = builder.q2i[1j]
        builder.circuit.append('CZ', [a, b, c, d])

    you would say

        builder.append('CZ', [(5+1j, 5), (0, 1j)])

    2) It canonicalizes. In particular, it will:
        - Sort targets. For example:
            `H 3 1 2` -> `H 1 2 3`
            `CX 2 3 1 0` -> `CX 1 0 2 3`
            `CZ 2 3 6 0` -> `CZ 0 6 2 3`
        - Replace rare gates with common gates. For example:
            `XCZ 1 2` -> `CX 2 1`
        - Not append target-less gates at all. For example:
            `CX      ` -> ``

        Canonicalization makes the form of the final circuit stable,
        despite things like python's `set` data structure having
        inconsistent iteration orders. This makes the output easier
        to unit test, and more viable to store under source control.

    3) It tracks measurements. When appending a measurement, its index is
    stored in the measurement tracker keyed by the position of the qubit
    being measured (or by a custom key, if `measure_key_func` is specified).
    The indices of the measurements can be looked up later via
    `builder.lookup_recs([key1, key2, ...])`.

    Args:
        gate: The name of the gate to append, such as "H" or "M" or "CX".
        targets: The qubit positions that the gate operates on. For single
            qubit gates like H or M this should be an iterable of complex
            numbers. For two qubit gates like CX or MXX it should be an
            iterable of pairs of complex numbers. For MPP it should be an
            iterable of gen.PauliMap instances.
        arg: Optional. The parens argument or arguments used for the gate
            instruction. For example, for a measurement gate, this is the
            probability of the incorrect result being reported.
        measure_key_func: Customizes the keys used to track the indices of
            measurement results. By default, measurements are keyed by
            position, but thus won't work if a circuit measures the same
            qubit multiple times. This function can transform that position
            into a different value (for example, you might set
            `measure_key_func=lambda pos: (pos, 'first_cycle')` for
            measurements during the first cycle of the circuit.
        tag: Defaults to "" (no tag). A custom tag to attach to the
            instruction(s) appended into the stim circuit.
    """
```

<a name="gen.ChunkBuilder.classical_paulis"></a>
```python
# gen.ChunkBuilder.classical_paulis

# (in class gen.ChunkBuilder)
def classical_paulis(
    self,
    *,
    control_keys: 'Iterable[Any]',
    targets: 'Iterable[complex]',
    basis: 'str',
) -> 'None':
    """Appends the tensor product of the given controls and targets into the circuit.
    """
```

<a name="gen.ChunkBuilder.demolition_measure_with_feedback_passthrough"></a>
```python
# gen.ChunkBuilder.demolition_measure_with_feedback_passthrough

# (in class gen.ChunkBuilder)
def demolition_measure_with_feedback_passthrough(
    self,
    xs: 'Iterable[complex]' = (),
    ys: 'Iterable[complex]' = (),
    zs: 'Iterable[complex]' = (),
    *,
    measure_key_func: 'Callable[[complex], Any]' = lambda e: e,
) -> 'None':
    """Performs demolition measurements that look like measurements w.r.t. detectors.

    This is done by adding feedback operations that flip the demolished qubits depending
    on the measurement result. This feedback can then later be removed using
    stim.Circuit.with_inlined_feedback. The benefit is that it can be easier to
    programmatically create the detectors using the passthrough measurements, and
    then they can be automatically converted.
    """
```

<a name="gen.ChunkBuilder.finish_chunk"></a>
```python
# gen.ChunkBuilder.finish_chunk

# (in class gen.ChunkBuilder)
def finish_chunk(
    self,
    *,
    wants_to_merge_with_prev: 'bool' = False,
    wants_to_merge_with_next: 'bool' = False,
    failure_mode: "Literal['error', 'ignore', 'print']" = 'error',
) -> 'Chunk':
```

<a name="gen.ChunkBuilder.finish_chunk_keep_semi_auto"></a>
```python
# gen.ChunkBuilder.finish_chunk_keep_semi_auto

# (in class gen.ChunkBuilder)
def finish_chunk_keep_semi_auto(
    self,
    *,
    discarded_inputs: 'Iterable[PauliMap]' = (),
    discarded_outputs: 'Iterable[PauliMap]' = (),
    wants_to_merge_with_prev: 'bool' = False,
    wants_to_merge_with_next: 'bool' = False,
) -> 'ChunkSemiAuto':
```

<a name="gen.ChunkBuilder.lookup_mids"></a>
```python
# gen.ChunkBuilder.lookup_mids

# (in class gen.ChunkBuilder)
def lookup_mids(
    self,
    keys: 'Iterable[Any]',
    *,
    ignore_unmatched: 'bool' = False,
) -> 'list[int]':
    """Looks up measurement indices by key.

    Measurement keys are created automatically when appending measurement operations into the
    circuit via the builder's append method. They are also created manually by methods like
    `builder.record_measurement_group`.

    Args:
        keys: The measurement keys to lookup.
        ignore_unmatched: Defaults to False. If set to True, keys that don't correspond
            to measurements are ignored instead of raising an error.

    Returns:
        A list of offsets indicating when the measurements occurred.
    """
```

<a name="gen.ChunkBuilder.record_measurement_group"></a>
```python
# gen.ChunkBuilder.record_measurement_group

# (in class gen.ChunkBuilder)
def record_measurement_group(
    self,
    sub_keys: 'Iterable[Any]',
    *,
    key: 'Any',
) -> 'None':
    """Combines multiple measurement keys into one key.

    Args:
        sub_keys: The measurement keys to combine.
        key: Where to store the combined result.
    """
```

<a name="gen.ChunkCompiler"></a>
```python
# gen.ChunkCompiler

# (at top-level in the gen module)
class ChunkCompiler:
    """Compiles appended chunks into a unified circuit.
    """
```

<a name="gen.ChunkCompiler.__init__"></a>
```python
# gen.ChunkCompiler.__init__

# (in class gen.ChunkCompiler)
def __init__(
    self,
    *,
    metadata_func: 'Callable[[Flow], FlowMetadata]' = lambda _: FlowMetadata(),
):
    """

    Args:
        metadata_func: Determines coordinate data appended to detectors
            (after x, y, and t).
    """
```

<a name="gen.ChunkCompiler.append"></a>
```python
# gen.ChunkCompiler.append

# (in class gen.ChunkCompiler)
def append(
    self,
    appended: 'Chunk | ChunkSemiAuto | ChunkLoop | ChunkReflow',
) -> 'None':
    """Appends a chunk to the circuit being built.

    The input flows of the appended chunk must exactly match the open outgoing flows of the
    circuit so far.
    """
```

<a name="gen.ChunkCompiler.append_magic_end_chunk"></a>
```python
# gen.ChunkCompiler.append_magic_end_chunk

# (in class gen.ChunkCompiler)
def append_magic_end_chunk(
    self,
    expected: 'ChunkInterface | None' = None,
) -> 'None':
    """Appends a non-physical chunk that cleanly terminates the circuit, regardless of open flows.

    Args:
        expected: Defaults to None (unused). If set to None, no extra checks are performed.
            If set to a ChunkInterface, it is verified that the open flows actually
            correspond to this interface.
    """
```

<a name="gen.ChunkCompiler.append_magic_init_chunk"></a>
```python
# gen.ChunkCompiler.append_magic_init_chunk

# (in class gen.ChunkCompiler)
def append_magic_init_chunk(
    self,
    expected: 'ChunkInterface | None' = None,
) -> 'None':
    """Appends a non-physical chunk that outputs the flows expected by the next chunk.

    Args:
        expected: Defaults to None (unused). If set to a ChunkInterface, it will be
            verified that the next appended chunk actually has a start interface
            matching the given expected interface. If set to None, then no checks are
            performed; no constraints are placed on the next chunk.
    """
```

<a name="gen.ChunkCompiler.copy"></a>
```python
# gen.ChunkCompiler.copy

# (in class gen.ChunkCompiler)
def copy(
    self,
) -> 'ChunkCompiler':
    """Returns a deep copy of the compiler's state.
    """
```

<a name="gen.ChunkCompiler.cur_circuit_html_viewer"></a>
```python
# gen.ChunkCompiler.cur_circuit_html_viewer

# (in class gen.ChunkCompiler)
def cur_circuit_html_viewer(
    self,
) -> 'gen.str_html':
```

<a name="gen.ChunkCompiler.cur_end_interface"></a>
```python
# gen.ChunkCompiler.cur_end_interface

# (in class gen.ChunkCompiler)
def cur_end_interface(
    self,
) -> 'ChunkInterface':
```

<a name="gen.ChunkCompiler.ensure_qubits_included"></a>
```python
# gen.ChunkCompiler.ensure_qubits_included

# (in class gen.ChunkCompiler)
def ensure_qubits_included(
    self,
    qubits: 'Iterable[complex]',
):
    """Adds the given qubit positions to the indexed positions, if they aren't already.
    """
```

<a name="gen.ChunkCompiler.finish_circuit"></a>
```python
# gen.ChunkCompiler.finish_circuit

# (in class gen.ChunkCompiler)
def finish_circuit(
    self,
) -> 'stim.Circuit':
    """Returns the circuit built by the compiler.

    Performs some final translation steps:
    - Re-indexing the qubits to be in a sorted order.
    - Re-indexing the observables to omit discarded observable flows.
    """
```

<a name="gen.ChunkInterface"></a>
```python
# gen.ChunkInterface

# (at top-level in the gen module)
class ChunkInterface:
    """Specifies a set of stabilizers and observables that a chunk can consume or prepare.
    """
```

<a name="gen.ChunkInterface.data_set"></a>
```python
# gen.ChunkInterface.data_set

# (in class gen.ChunkInterface)
class data_set:
```

<a name="gen.ChunkInterface.partitioned_detector_flows"></a>
```python
# gen.ChunkInterface.partitioned_detector_flows

# (in class gen.ChunkInterface)
def partitioned_detector_flows(
    self,
) -> 'list[list[PauliMap]]':
    """Returns the stabilizers of the interface, split into non-overlapping groups.
    """
```

<a name="gen.ChunkInterface.to_code"></a>
```python
# gen.ChunkInterface.to_code

# (in class gen.ChunkInterface)
def to_code(
    self,
) -> 'StabilizerCode':
    """Returns a gen.StabilizerCode with stabilizers/logicals determined by what the chunk interface mentions.
    """
```

<a name="gen.ChunkInterface.to_patch"></a>
```python
# gen.ChunkInterface.to_patch

# (in class gen.ChunkInterface)
def to_patch(
    self,
) -> 'Patch':
    """Returns a gen.Patch with tiles equal to the chunk interface's stabilizers.
    """
```

<a name="gen.ChunkInterface.to_svg"></a>
```python
# gen.ChunkInterface.to_svg

# (in class gen.ChunkInterface)
def to_svg(
    self,
    *,
    show_order: 'bool' = False,
    show_measure_qubits: 'bool' = False,
    show_data_qubits: 'bool' = True,
    system_qubits: 'Iterable[complex]' = (),
    opacity: 'float' = 1,
    show_coords: 'bool' = True,
    show_obs: 'bool' = True,
    other: 'StabilizerCode | Patch | Iterable[StabilizerCode | Patch] | None' = None,
    tile_color_func: 'Callable[[Tile], str] | None' = None,
    rows: 'int | None' = None,
    cols: 'int | None' = None,
    find_logical_err_max_weight: 'int | None' = None,
) -> 'str_svg':
```

<a name="gen.ChunkInterface.used_set"></a>
```python
# gen.ChunkInterface.used_set

# (in class gen.ChunkInterface)
class used_set:
    """Returns the set of qubits used in any flow mentioned by the chunk interface.
    """
```

<a name="gen.ChunkInterface.with_discards_as_ports"></a>
```python
# gen.ChunkInterface.with_discards_as_ports

# (in class gen.ChunkInterface)
def with_discards_as_ports(
    self,
) -> 'ChunkInterface':
    """Returns the same chunk interface, but with discarded flows turned into normal flows.
    """
```

<a name="gen.ChunkInterface.with_edits"></a>
```python
# gen.ChunkInterface.with_edits

# (in class gen.ChunkInterface)
def with_edits(
    self,
    *,
    ports: 'Iterable[PauliMap] | None' = None,
    discards: 'Iterable[PauliMap] | None' = None,
) -> 'ChunkInterface':
    """Returns an equivalent chunk interface but with the given values replaced.
    """
```

<a name="gen.ChunkInterface.with_transformed_coords"></a>
```python
# gen.ChunkInterface.with_transformed_coords

# (in class gen.ChunkInterface)
def with_transformed_coords(
    self,
    transform: 'Callable[[complex], complex]',
) -> 'ChunkInterface':
    """Returns the same chunk interface, but with coordinates transformed by the given function.
    """
```

<a name="gen.ChunkInterface.without_discards"></a>
```python
# gen.ChunkInterface.without_discards

# (in class gen.ChunkInterface)
def without_discards(
    self,
) -> 'ChunkInterface':
    """Returns the same chunk interface, but with discarded flows not included.
    """
```

<a name="gen.ChunkInterface.without_keyed"></a>
```python
# gen.ChunkInterface.without_keyed

# (in class gen.ChunkInterface)
def without_keyed(
    self,
) -> 'ChunkInterface':
    """Returns the same chunk interface, but without logical flows (named flows).
    """
```

<a name="gen.ChunkLoop"></a>
```python
# gen.ChunkLoop

# (at top-level in the gen module)
class ChunkLoop:
    """Specifies a series of chunks to repeat a fixed number of times.

    The loop invariant is that the last chunk's end interface should match the
    first chunk's start interface (unless the number of repetitions is less than
    2).

    For duck typing purposes, many methods supported by Chunk are supported by
    ChunkLoop.
    """
```

<a name="gen.ChunkLoop.end_interface"></a>
```python
# gen.ChunkLoop.end_interface

# (in class gen.ChunkLoop)
def end_interface(
    self,
) -> 'ChunkInterface':
    """Returns the end interface of the last chunk in the loop.
    """
```

<a name="gen.ChunkLoop.end_patch"></a>
```python
# gen.ChunkLoop.end_patch

# (in class gen.ChunkLoop)
def end_patch(
    self,
) -> 'Patch':
```

<a name="gen.ChunkLoop.find_distance"></a>
```python
# gen.ChunkLoop.find_distance

# (in class gen.ChunkLoop)
def find_distance(
    self,
    *,
    max_search_weight: 'int',
    noise: 'float | NoiseModel' = 0.001,
    noiseless_qubits: 'Iterable[float | int | complex]' = (),
) -> 'int':
```

<a name="gen.ChunkLoop.find_logical_error"></a>
```python
# gen.ChunkLoop.find_logical_error

# (in class gen.ChunkLoop)
def find_logical_error(
    self,
    *,
    max_search_weight: 'int',
    noise: 'float | NoiseModel' = 0.001,
    noiseless_qubits: 'Iterable[float | int | complex]' = (),
) -> 'list[stim.ExplainedError]':
    """Searches for a minium distance undetected logical error.

    By default, searches using a uniform depolarizing circuit noise model.
    """
```

<a name="gen.ChunkLoop.flattened"></a>
```python
# gen.ChunkLoop.flattened

# (in class gen.ChunkLoop)
def flattened(
    self,
) -> 'list[Chunk | ChunkReflow]':
    """Unrolls the loop, and any sub-loops, into a series of chunks.
    """
```

<a name="gen.ChunkLoop.start_interface"></a>
```python
# gen.ChunkLoop.start_interface

# (in class gen.ChunkLoop)
def start_interface(
    self,
) -> 'ChunkInterface':
    """Returns the start interface of the first chunk in the loop.
    """
```

<a name="gen.ChunkLoop.start_patch"></a>
```python
# gen.ChunkLoop.start_patch

# (in class gen.ChunkLoop)
def start_patch(
    self,
) -> 'Patch':
```

<a name="gen.ChunkLoop.time_reversed"></a>
```python
# gen.ChunkLoop.time_reversed

# (in class gen.ChunkLoop)
def time_reversed(
    self,
) -> 'ChunkLoop':
    """Returns the same loop, but time reversed.

    The time reversed loop has reversed flows, implemented by performs operations in the
    reverse order and exchange measurements for resets (and vice versa) as appropriate.
    It has exactly the same fault tolerant structure, just mirrored in time.
    """
```

<a name="gen.ChunkLoop.to_closed_circuit"></a>
```python
# gen.ChunkLoop.to_closed_circuit

# (in class gen.ChunkLoop)
def to_closed_circuit(
    self,
) -> 'stim.Circuit':
    """Compiles the chunk into a circuit by conjugating with mpp init/end chunks.
    """
```

<a name="gen.ChunkLoop.to_html_viewer"></a>
```python
# gen.ChunkLoop.to_html_viewer

# (in class gen.ChunkLoop)
def to_html_viewer(
    self,
    *,
    patch: 'Patch | StabilizerCode | ChunkInterface | None' = None,
    tile_color_func: 'Callable[[Tile], tuple[float, float, float, float]] | None' = None,
    known_error: 'Iterable[stim.ExplainedError] | None' = None,
) -> 'str_html':
    """Returns an HTML document containing a viewer for the chunk loop's circuit.
    """
```

<a name="gen.ChunkLoop.verify"></a>
```python
# gen.ChunkLoop.verify

# (in class gen.ChunkLoop)
def verify(
    self,
    *,
    expected_in: 'ChunkInterface | None' = None,
    expected_out: 'ChunkInterface | None' = None,
):
```

<a name="gen.ChunkLoop.verify_distance_is_at_least_2"></a>
```python
# gen.ChunkLoop.verify_distance_is_at_least_2

# (in class gen.ChunkLoop)
def verify_distance_is_at_least_2(
    self,
    *,
    noise: 'float | NoiseModel' = 0.001,
):
    """Verifies undetected logical errors require at least 2 physical errors.

    Verifies using a uniform depolarizing circuit noise model.
    """
```

<a name="gen.ChunkLoop.verify_distance_is_at_least_3"></a>
```python
# gen.ChunkLoop.verify_distance_is_at_least_3

# (in class gen.ChunkLoop)
def verify_distance_is_at_least_3(
    self,
    *,
    noise: 'float | NoiseModel' = 0.001,
):
    """Verifies undetected logical errors require at least 3 physical errors.

    By default, verifies using a uniform depolarizing circuit noise model.
    """
```

<a name="gen.ChunkLoop.with_repetitions"></a>
```python
# gen.ChunkLoop.with_repetitions

# (in class gen.ChunkLoop)
def with_repetitions(
    self,
    new_repetitions: 'int',
) -> 'ChunkLoop':
    """Returns the same loop, but with a different number of repetitions.
    """
```

<a name="gen.ChunkReflow"></a>
```python
# gen.ChunkReflow

# (at top-level in the gen module)
class ChunkReflow:
    """An adapter chunk for attaching chunks describing the same thing in different ways.

    For example, consider two surface code idle round chunks where one has the logical
    operator on the left side and the other has the logical operator on the right side.
    They can't be directly concatenated, because their flows don't match. But a reflow
    chunk can be placed in between, mapping the left logical operator to the right
    logical operator times a set of stabilizers, in order to bridge the incompatibility.
    """
```

<a name="gen.ChunkReflow.end_code"></a>
```python
# gen.ChunkReflow.end_code

# (in class gen.ChunkReflow)
def end_code(
    self,
) -> 'StabilizerCode':
```

<a name="gen.ChunkReflow.end_interface"></a>
```python
# gen.ChunkReflow.end_interface

# (in class gen.ChunkReflow)
def end_interface(
    self,
) -> 'ChunkInterface':
```

<a name="gen.ChunkReflow.end_patch"></a>
```python
# gen.ChunkReflow.end_patch

# (in class gen.ChunkReflow)
def end_patch(
    self,
) -> 'Patch':
```

<a name="gen.ChunkReflow.flattened"></a>
```python
# gen.ChunkReflow.flattened

# (in class gen.ChunkReflow)
def flattened(
    self,
) -> 'list[ChunkReflow]':
    """This is here for duck-type compatibility with ChunkLoop.
    """
```

<a name="gen.ChunkReflow.from_auto_rewrite"></a>
```python
# gen.ChunkReflow.from_auto_rewrite

# (in class gen.ChunkReflow)
def from_auto_rewrite(
    *,
    inputs: 'Iterable[PauliMap]',
    out2in: "dict[PauliMap, list[PauliMap] | Literal['auto']]",
) -> 'ChunkReflow':
```

<a name="gen.ChunkReflow.from_auto_rewrite_transitions_using_stable"></a>
```python
# gen.ChunkReflow.from_auto_rewrite_transitions_using_stable

# (in class gen.ChunkReflow)
def from_auto_rewrite_transitions_using_stable(
    *,
    stable: 'Iterable[PauliMap]',
    transitions: 'Iterable[tuple[PauliMap, PauliMap]]',
) -> 'ChunkReflow':
    """Bridges the given transitions using products from the given stable values.
    """
```

<a name="gen.ChunkReflow.removed_inputs"></a>
```python
# gen.ChunkReflow.removed_inputs

# (in class gen.ChunkReflow)
class removed_inputs:
```

<a name="gen.ChunkReflow.start_code"></a>
```python
# gen.ChunkReflow.start_code

# (in class gen.ChunkReflow)
def start_code(
    self,
) -> 'StabilizerCode':
```

<a name="gen.ChunkReflow.start_interface"></a>
```python
# gen.ChunkReflow.start_interface

# (in class gen.ChunkReflow)
def start_interface(
    self,
) -> 'ChunkInterface':
```

<a name="gen.ChunkReflow.start_patch"></a>
```python
# gen.ChunkReflow.start_patch

# (in class gen.ChunkReflow)
def start_patch(
    self,
) -> 'Patch':
```

<a name="gen.ChunkReflow.verify"></a>
```python
# gen.ChunkReflow.verify

# (in class gen.ChunkReflow)
def verify(
    self,
    *,
    expected_in: 'StabilizerCode | ChunkInterface | None' = None,
    expected_out: 'StabilizerCode | ChunkInterface | None' = None,
):
    """Verifies that the ChunkReflow is well-formed.
    """
```

<a name="gen.ChunkReflow.with_obs_flows_as_det_flows"></a>
```python
# gen.ChunkReflow.with_obs_flows_as_det_flows

# (in class gen.ChunkReflow)
def with_obs_flows_as_det_flows(
    self,
):
```

<a name="gen.ChunkReflow.with_transformed_coords"></a>
```python
# gen.ChunkReflow.with_transformed_coords

# (in class gen.ChunkReflow)
def with_transformed_coords(
    self,
    transform: 'Callable[[complex], complex]',
) -> 'ChunkReflow':
```

<a name="gen.ChunkSemiAuto"></a>
```python
# gen.ChunkSemiAuto

# (at top-level in the gen module)
class ChunkSemiAuto:
    """A variant of `gen.Chunk` that supports partially specified flows.

    Use `gen.ChunkSemiAuto.solve()` to solve the partially specified flows and
    return a solved `gen.Chunk`.
    """
```

<a name="gen.ChunkSemiAuto.__init__"></a>
```python
# gen.ChunkSemiAuto.__init__

# (in class gen.ChunkSemiAuto)
def __init__(
    self,
    circuit: 'stim.Circuit',
    *,
    flows: 'Iterable[FlowSemiAuto | Flow]',
    discarded_inputs: 'Iterable[PauliMap | Tile]' = (),
    discarded_outputs: 'Iterable[PauliMap | Tile]' = (),
    wants_to_merge_with_next: 'bool' = False,
    wants_to_merge_with_prev: 'bool' = False,
    q2i: 'dict[complex, int] | None' = None,
    o2i: 'dict[Any, int] | None' = None,
):
    """

    Args:
        circuit: The circuit implementing the chunk's functionality.
        flows: A series of stabilizer flows that the circuit implements.
        discarded_inputs: Explicitly rejected in flows. For example, a data
            measurement chunk might reject flows for stabilizers from the
            anticommuting basis. If they are not rejected, then compilation
            will fail when attempting to combine this chunk with a preceding
            chunk that has those stabilizers from the anticommuting basis
            flowing out.
        discarded_outputs: Explicitly rejected out flows. For example, an
            initialization chunk might reject flows for stabilizers from the
            anticommuting basis. If they are not rejected, then compilation
            will fail when attempting to combine this chunk with a following
            chunk that has those stabilizers from the anticommuting basis
            flowing in.
        wants_to_merge_with_next: Defaults to False. When set to True,
            the chunk compiler won't insert a TICK between this chunk
            and the next chunk.
        wants_to_merge_with_prev: Defaults to False. When set to True,
            the chunk compiler won't insert a TICK between this chunk
            and the previous chunk.
        q2i: Defaults to None (infer from QUBIT_COORDS instructions in circuit else
            raise an exception). The gen-qubit-coordinate-to-stim-qubit-index mapping
            used to translate between gen's qubit keys and stim's qubit keys.
        o2i: Defaults to None (raise an exception if observables present in circuit).
            The gen-observable-key-to-stim-observable-index mapping used to translate
            between gen's observable keys and stim's observable keys.
    """
```

<a name="gen.ChunkSemiAuto.solve"></a>
```python
# gen.ChunkSemiAuto.solve

# (in class gen.ChunkSemiAuto)
def solve(
    self,
    *,
    failure_mode: "Literal['error', 'ignore', 'print']" = 'error',
) -> 'Chunk':
    """Solves any partially specified flows, and returns a `gen.Chunk` with the solution.
    """
```

<a name="gen.ColoredLineData"></a>
```python
# gen.ColoredLineData

# (at top-level in the gen module)
class ColoredLineData:
```

<a name="gen.ColoredLineData.fused"></a>
```python
# gen.ColoredLineData.fused

# (in class gen.ColoredLineData)
def fused(
    data: 'Iterable[ColoredLineData]',
) -> 'list[ColoredLineData]':
```

<a name="gen.ColoredTriangleData"></a>
```python
# gen.ColoredTriangleData

# (at top-level in the gen module)
class ColoredTriangleData:
```

<a name="gen.ColoredTriangleData.fused"></a>
```python
# gen.ColoredTriangleData.fused

# (in class gen.ColoredTriangleData)
def fused(
    data: 'Iterable[ColoredTriangleData]',
) -> 'list[ColoredTriangleData]':
```

<a name="gen.ColoredTriangleData.square"></a>
```python
# gen.ColoredTriangleData.square

# (in class gen.ColoredTriangleData)
def square(
    *,
    rgba: 'tuple[float, float, float, float]',
    origin: 'Iterable[float]',
    d1: 'Iterable[float]',
    d2: 'Iterable[float]',
) -> 'ColoredTriangleData':
```

<a name="gen.Flow"></a>
```python
# gen.Flow

# (at top-level in the gen module)
class Flow:
    """A rule for how a stabilizer travels into, through, and/or out of a circuit.
    """
```

<a name="gen.Flow.__init__"></a>
```python
# gen.Flow.__init__

# (in class gen.Flow)
def __init__(
    self,
    *,
    start: 'PauliMap | Tile | None' = None,
    end: 'PauliMap | Tile | None' = None,
    mids: 'Iterable[int]' = (),
    obs_key: 'Any' = None,
    center: 'complex | None' = None,
    flags: 'Iterable[str]' = frozenset(),
    sign: 'bool | None' = None,
):
    """Initializes a Flow.

    Args:
        start: Defaults to None (empty). The Pauli product operator at the beginning of the
            circuit (before *all* operations, including resets).
        end: Defaults to None (empty). The Pauli product operator at the end of the
            circuit (after *all* operations, including measurements).
        mids: Defaults to empty. Indices of measurements that mediate the flow (that multiply
            into it as it traverses the circuit).
        center: Defaults to None (unspecified). Specifies a 2d coordinate to use in metadata
            when the flow is completed into a detector. Incompatible with obs_key.
        obs_key: Defaults to None (detector flow). Identifies that this is an observable flow
            (instead of a detector flow) and gives a name that be used when linking chunks.
        flags: Defaults to empty. Custom information about the flow, that can be used by code
            operating on chunks for a variety of purposes. For example, this could identify the
            "color" of the flow in a color code.
        sign: Defaults to None (unsigned). The expected sign of the flow.
    """
```

<a name="gen.Flow.fuse_with_next_flow"></a>
```python
# gen.Flow.fuse_with_next_flow

# (in class gen.Flow)
def fuse_with_next_flow(
    self,
    next_flow: 'Flow',
    *,
    next_flow_measure_offset: 'int',
) -> 'Flow':
```

<a name="gen.Flow.key_end"></a>
```python
# gen.Flow.key_end

# (in class gen.Flow)
@property
def key_end(
    self,
):
```

<a name="gen.Flow.key_start"></a>
```python
# gen.Flow.key_start

# (in class gen.Flow)
@property
def key_start(
    self,
):
```

<a name="gen.Flow.to_stim_flow"></a>
```python
# gen.Flow.to_stim_flow

# (in class gen.Flow)
def to_stim_flow(
    self,
    *,
    q2i: 'dict[complex, int]',
    o2i: 'dict[Any, int] | None' = None,
) -> 'stim.Flow':
```

<a name="gen.Flow.with_edits"></a>
```python
# gen.Flow.with_edits

# (in class gen.Flow)
def with_edits(
    self,
    *,
    start: 'PauliMap | None' = None,
    end: 'PauliMap | None' = None,
    measurement_indices: 'Iterable[int] | None' = None,
    obs_key: 'Any' = '__not_specified!!',
    center: 'complex | None' = None,
    flags: 'Iterable[str] | None' = None,
    sign: 'Any' = '__not_specified!!',
) -> 'Flow':
```

<a name="gen.Flow.with_transformed_coords"></a>
```python
# gen.Flow.with_transformed_coords

# (in class gen.Flow)
def with_transformed_coords(
    self,
    transform: 'Callable[[complex], complex]',
) -> 'Flow':
```

<a name="gen.Flow.with_xz_flipped"></a>
```python
# gen.Flow.with_xz_flipped

# (in class gen.Flow)
def with_xz_flipped(
    self,
) -> 'Flow':
```

<a name="gen.FlowMetadata"></a>
```python
# gen.FlowMetadata

# (at top-level in the gen module)
class FlowMetadata:
    """Metadata, based on a flow, to use during circuit generation.
    """
```

<a name="gen.FlowMetadata.__init__"></a>
```python
# gen.FlowMetadata.__init__

# (in class gen.FlowMetadata)
def __init__(
    self,
    *,
    extra_coords: 'Iterable[float]' = (),
    tag: 'str | None' = '',
):
    """
    Args:
        extra_coords: Extra numbers to add to DETECTOR coordinate arguments. By default gen
            gives each detector an X, Y, and T coordinate. These numbers go afterward.
        tag: A tag to attach to DETECTOR or OBSERVABLE_INCLUDE instructions.
    """
```

<a name="gen.FlowSemiAuto"></a>
```python
# gen.FlowSemiAuto

# (at top-level in the gen module)
class FlowSemiAuto:
    """A rule for how a stabilizer travels into, through, and/or out of a circuit.
    """
```

<a name="gen.FlowSemiAuto.__init__"></a>
```python
# gen.FlowSemiAuto.__init__

# (in class gen.FlowSemiAuto)
def __init__(
    self,
    *,
    start: "PauliMap | Tile | Literal['auto'] | None" = None,
    end: "PauliMap | Tile | Literal['auto'] | None" = None,
    mids: "Iterable[int] | Literal['auto']" = (),
    obs_key: 'Any' = None,
    center: 'complex | None' = None,
    flags: 'Iterable[str]' = frozenset(),
    sign: 'bool | None' = None,
):
    """Initializes a Flow.

    Args:
        start: Defaults to None (empty). The Pauli product operator at the beginning of the
            circuit (before *all* operations, including resets).
        end: Defaults to None (empty). The Pauli product operator at the end of the
            circuit (after *all* operations, including measurements).
        mids: Defaults to empty. Indices of measurements that mediate the flow (that multiply
            into it as it traverses the circuit).
        center: Defaults to None (auto). Specifies a 2d coordinate to use in metadata when the
            flow is completed into a detector. Incompatible with obs_key. Derived automatically
            when not specified.
        obs_key: Defaults to None (detector flow). Identifies that this is an observable flow
            (instead of a detector flow) and gives a name that be used when linking chunks.
        flags: Defaults to empty. Custom information about the flow, that can be used by code
            operating on chunks for a variety of purposes. For example, this could identify the
            "color" of the flow in a color code.
        sign: Defaults to None (unsigned). The expected sign of the flow.
    """
```

<a name="gen.FlowSemiAuto.to_flow"></a>
```python
# gen.FlowSemiAuto.to_flow

# (in class gen.FlowSemiAuto)
def to_flow(
    self,
) -> 'Flow':
    """Converts the solved FlowSemiAuto to a Flow.

    If there are still 'auto' fields present, the conversion fails.
    """
```

<a name="gen.FlowSemiAuto.to_stim_flow"></a>
```python
# gen.FlowSemiAuto.to_stim_flow

# (in class gen.FlowSemiAuto)
def to_stim_flow(
    self,
    *,
    q2i: 'dict[complex, int]',
    o2i: 'dict[Any, int] | None' = None,
) -> 'stim.Flow':
```

<a name="gen.FlowSemiAuto.with_edits"></a>
```python
# gen.FlowSemiAuto.with_edits

# (in class gen.FlowSemiAuto)
def with_edits(
    self,
    *,
    start: "PauliMap | Literal['auto'] | None" = None,
    end: "PauliMap | Literal['auto'] | None" = None,
    measurement_indices: "Iterable[int] | Literal['auto'] | None" = None,
    obs_key: 'Any' = '__not_specified!!',
    center: 'complex | None' = None,
    flags: 'Iterable[str] | None' = None,
    sign: 'Any' = '__not_specified!!',
) -> 'FlowSemiAuto':
```

<a name="gen.InteractLayer"></a>
```python
# gen.InteractLayer

# (at top-level in the gen module)
@dataclasses.dataclass
class InteractLayer:
    """A layer of controlled Pauli gates (like CX, CZ, and XCY).
    """
    targets1: list[int]
    targets2: list[int]
    bases1: list[str]
    bases2: list[str]
```

<a name="gen.InteractLayer.append_into_stim_circuit"></a>
```python
# gen.InteractLayer.append_into_stim_circuit

# (in class gen.InteractLayer)
def append_into_stim_circuit(
    self,
    out: 'stim.Circuit',
) -> 'None':
```

<a name="gen.InteractLayer.copy"></a>
```python
# gen.InteractLayer.copy

# (in class gen.InteractLayer)
def copy(
    self,
) -> 'InteractLayer':
```

<a name="gen.InteractLayer.locally_optimized"></a>
```python
# gen.InteractLayer.locally_optimized

# (in class gen.InteractLayer)
def locally_optimized(
    self,
    next_layer: 'Layer | None',
) -> 'list[Layer | None]':
```

<a name="gen.InteractLayer.rotate_to_z_layer"></a>
```python
# gen.InteractLayer.rotate_to_z_layer

# (in class gen.InteractLayer)
def rotate_to_z_layer(
    self,
):
```

<a name="gen.InteractLayer.to_z_basis"></a>
```python
# gen.InteractLayer.to_z_basis

# (in class gen.InteractLayer)
def to_z_basis(
    self,
) -> 'list[Layer]':
```

<a name="gen.InteractLayer.touched"></a>
```python
# gen.InteractLayer.touched

# (in class gen.InteractLayer)
def touched(
    self,
) -> 'set[int]':
```

<a name="gen.LayerCircuit"></a>
```python
# gen.LayerCircuit

# (at top-level in the gen module)
@dataclasses.dataclass
class LayerCircuit:
    """A stabilizer circuit represented as a series of typed layers.

    For example, the circuit could be a `ResetLayer`, then a `RotationLayer`,
    then a few `InteractLayer`s, then a `MeasureLayer`.
    """
    layers: list[Layer]
```

<a name="gen.LayerCircuit.copy"></a>
```python
# gen.LayerCircuit.copy

# (in class gen.LayerCircuit)
def copy(
    self,
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.from_stim_circuit"></a>
```python
# gen.LayerCircuit.from_stim_circuit

# (in class gen.LayerCircuit)
def from_stim_circuit(
    circuit: 'stim.Circuit',
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.to_stim_circuit"></a>
```python
# gen.LayerCircuit.to_stim_circuit

# (in class gen.LayerCircuit)
def to_stim_circuit(
    self,
) -> 'stim.Circuit':
    """Compiles the layer circuit into a stim circuit and returns it.
    """
```

<a name="gen.LayerCircuit.to_z_basis"></a>
```python
# gen.LayerCircuit.to_z_basis

# (in class gen.LayerCircuit)
def to_z_basis(
    self,
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.touched"></a>
```python
# gen.LayerCircuit.touched

# (in class gen.LayerCircuit)
def touched(
    self,
) -> 'set[int]':
```

<a name="gen.LayerCircuit.with_cleaned_up_loop_iterations"></a>
```python
# gen.LayerCircuit.with_cleaned_up_loop_iterations

# (in class gen.LayerCircuit)
def with_cleaned_up_loop_iterations(
    self,
) -> 'LayerCircuit':
    """Attempts to roll up partially unrolled loops.

    Checks if the instructions before a loop correspond to the instruction inside a loop. If so,
    removes the matching instructions beforehand and increases the iteration count by 1. Same
    for instructions after the loop.

    This essentially undoes the effect of `with_ejected_loop_iterations`. A common pattern is
    to do `with_ejected_loop_iterations`, then an optimization, then
    `with_cleaned_up_loop_iterations`. This gives the optimization the chance to optimize across
    a loop boundary, but cleans up after itself if no optimization occurs.

    In some cases this method is useful because of circuit generation code being overly cautious
    about how quickly loop invariants are established, and so emitting the first iteration of a
    loop in a special way. If it happens to be identical, despite the different code path that
    produced it, this method will roll it into the rest of the loop.

    For example, this method would turn this circuit fragment:

        X 0
        MR 0
        REPEAT 98 {
            X 0
            MR 0
        }
        X 0
        MR 0

    into this circuit fragment:

        REPEAT 100 {
            X 0
            MR 0
        }
    """
```

<a name="gen.LayerCircuit.with_clearable_rotation_layers_cleared"></a>
```python
# gen.LayerCircuit.with_clearable_rotation_layers_cleared

# (in class gen.LayerCircuit)
def with_clearable_rotation_layers_cleared(
    self,
) -> 'LayerCircuit':
    """Removes rotation layers where every rotation in the layer can be moved to another layer.

    Each individual rotation can move through intermediate non-rotation layers as long as those
    layers don't touch the qubit being rotated.
    """
```

<a name="gen.LayerCircuit.with_ejected_loop_iterations"></a>
```python
# gen.LayerCircuit.with_ejected_loop_iterations

# (in class gen.LayerCircuit)
def with_ejected_loop_iterations(
    self,
) -> 'LayerCircuit':
    """Partially unrolls loops, placing one iteration before and one iteration after.

    This is useful for ensuring the transition into and out of a loop is optimized correctly.
    For example, if a circuit begins with a transversal initialization of data qubits and then
    immediately starts a memory loop, the resets from the data initialization should be merged
    into the same layer as the resets from the measurement initialization at the beginning of
    the loop. But the reset-merging optimization might not see that this is possible across the
    loop boundary. Ejecting an iteration fixes this issue.

    For example, this method would turn this circuit fragment:

        REPEAT 100 {
            X 0
            MR 0
        }

    into this circuit fragment:

        X 0
        MR 0
        REPEAT 98 {
            X 0
            MR 0
        }
        X 0
        MR 0
    """
```

<a name="gen.LayerCircuit.with_irrelevant_tail_layers_removed"></a>
```python
# gen.LayerCircuit.with_irrelevant_tail_layers_removed

# (in class gen.LayerCircuit)
def with_irrelevant_tail_layers_removed(
    self,
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.with_locally_merged_measure_layers"></a>
```python
# gen.LayerCircuit.with_locally_merged_measure_layers

# (in class gen.LayerCircuit)
def with_locally_merged_measure_layers(
    self,
) -> 'LayerCircuit':
    """Merges measurement layers together, despite intervening annotation layers.

    For example, this method would turn this circuit fragment:

        M 0
        DETECTOR(0, 0) rec[-1]
        OBSERVABLE_INCLUDE(5) rec[-1]
        SHIFT_COORDS(0, 1)
        M 1
        DETECTOR(0, 0) rec[-1]

    into this circuit fragment:

        M 0 1
        DETECTOR(0, 0) rec[-2]
        OBSERVABLE_INCLUDE(5) rec[-2]
        SHIFT_COORDS(0, 1)
        DETECTOR(0, 0) rec[-1]
    """
```

<a name="gen.LayerCircuit.with_locally_optimized_layers"></a>
```python
# gen.LayerCircuit.with_locally_optimized_layers

# (in class gen.LayerCircuit)
def with_locally_optimized_layers(
    self,
) -> 'LayerCircuit':
    """Iterates over the circuit aggregating layer.optimized(second_layer).
    """
```

<a name="gen.LayerCircuit.with_qubit_coords_at_start"></a>
```python
# gen.LayerCircuit.with_qubit_coords_at_start

# (in class gen.LayerCircuit)
def with_qubit_coords_at_start(
    self,
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.with_rotations_before_resets_removed"></a>
```python
# gen.LayerCircuit.with_rotations_before_resets_removed

# (in class gen.LayerCircuit)
def with_rotations_before_resets_removed(
    self,
    loop_boundary_resets: 'set[int] | None' = None,
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.with_rotations_merged_earlier"></a>
```python
# gen.LayerCircuit.with_rotations_merged_earlier

# (in class gen.LayerCircuit)
def with_rotations_merged_earlier(
    self,
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.with_rotations_rolled_from_end_of_loop_to_start_of_loop"></a>
```python
# gen.LayerCircuit.with_rotations_rolled_from_end_of_loop_to_start_of_loop

# (in class gen.LayerCircuit)
def with_rotations_rolled_from_end_of_loop_to_start_of_loop(
    self,
) -> 'LayerCircuit':
    """Rewrites loops so that they only have rotations at the start, not the end.

    This is useful for ensuring loops don't redundantly rotate at the loop boundary,
    by merging the rotations at the end with the rotations at the start or by
    making it clear rotations at the end were not needed because of the
    operations coming next.

    For example, this:

        REPEAT 5 {
            S 2 3 4
            R 0 1
            ...
            M 0 1
            H 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
            DETECTOR rec[-1]
        }

    will become this:

        REPEAT 5 {
            H 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
            S 2 3 4
            R 0 1
            ...
            M 0 1
            DETECTOR rec[-1]
        }

    which later optimization passes can then reduce further.
    """
```

<a name="gen.LayerCircuit.with_whole_layers_slid_as_early_as_possible_for_merge_with_same_layer"></a>
```python
# gen.LayerCircuit.with_whole_layers_slid_as_early_as_possible_for_merge_with_same_layer

# (in class gen.LayerCircuit)
def with_whole_layers_slid_as_early_as_possible_for_merge_with_same_layer(
    self,
    layer_types: 'type | tuple[type, ...]',
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.with_whole_layers_slid_as_to_merge_with_previous_layer_of_same_type"></a>
```python
# gen.LayerCircuit.with_whole_layers_slid_as_to_merge_with_previous_layer_of_same_type

# (in class gen.LayerCircuit)
def with_whole_layers_slid_as_to_merge_with_previous_layer_of_same_type(
    self,
    layer_types: 'type | tuple[type, ...]',
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.with_whole_rotation_layers_slid_earlier"></a>
```python
# gen.LayerCircuit.with_whole_rotation_layers_slid_earlier

# (in class gen.LayerCircuit)
def with_whole_rotation_layers_slid_earlier(
    self,
) -> 'LayerCircuit':
```

<a name="gen.LayerCircuit.without_empty_layers"></a>
```python
# gen.LayerCircuit.without_empty_layers

# (in class gen.LayerCircuit)
def without_empty_layers(
    self,
) -> 'LayerCircuit':
    """Removes empty layers from the circuit.

    Empty layers are sometimes created as a byproduct of certain optimizations, or may have been
    present in the original circuit. Usually they are unwanted, and this method removes them.
    """
```

<a name="gen.MeasureLayer"></a>
```python
# gen.MeasureLayer

# (at top-level in the gen module)
@dataclasses.dataclass
class MeasureLayer:
    """A layer of single qubit Pauli basis measurement operations.
    """
    targets: list[int]
    bases: list[str]
```

<a name="gen.MeasureLayer.append_into_stim_circuit"></a>
```python
# gen.MeasureLayer.append_into_stim_circuit

# (in class gen.MeasureLayer)
def append_into_stim_circuit(
    self,
    out: 'stim.Circuit',
) -> 'None':
```

<a name="gen.MeasureLayer.copy"></a>
```python
# gen.MeasureLayer.copy

# (in class gen.MeasureLayer)
def copy(
    self,
) -> 'MeasureLayer':
```

<a name="gen.MeasureLayer.locally_optimized"></a>
```python
# gen.MeasureLayer.locally_optimized

# (in class gen.MeasureLayer)
def locally_optimized(
    self,
    next_layer: 'Layer | None',
) -> 'list[Layer | None]':
```

<a name="gen.MeasureLayer.to_z_basis"></a>
```python
# gen.MeasureLayer.to_z_basis

# (in class gen.MeasureLayer)
def to_z_basis(
    self,
) -> 'list[Layer]':
```

<a name="gen.MeasureLayer.touched"></a>
```python
# gen.MeasureLayer.touched

# (in class gen.MeasureLayer)
def touched(
    self,
) -> 'set[int]':
```

<a name="gen.NoiseModel"></a>
```python
# gen.NoiseModel

# (at top-level in the gen module)
class NoiseModel:
    """Converts circuits into noisy circuits according to rules.
    """
```

<a name="gen.NoiseModel.noisy_circuit"></a>
```python
# gen.NoiseModel.noisy_circuit

# (in class gen.NoiseModel)
def noisy_circuit(
    self,
    circuit: 'stim.Circuit',
    *,
    system_qubit_indices: 'set[int] | None' = None,
    immune_qubit_indices: 'Iterable[int] | None' = None,
    immune_qubit_coords: 'Iterable[complex | float | int | Iterable[float | int]] | None' = None,
) -> 'stim.Circuit':
    """Returns a noisy version of the given circuit, by applying the receiving noise model.

    Args:
        circuit: The circuit to layer noise over.
        system_qubit_indices: All qubits used by the circuit. These are the qubits eligible for
            idling noise.
        immune_qubit_indices: Qubits to not apply noise to, even if they are operated on.
        immune_qubit_coords: Qubit coordinates to not apply noise to, even if they are operated
            on.

    Returns:
        The noisy version of the circuit.
    """
```

<a name="gen.NoiseModel.noisy_circuit_skipping_mpp_boundaries"></a>
```python
# gen.NoiseModel.noisy_circuit_skipping_mpp_boundaries

# (in class gen.NoiseModel)
def noisy_circuit_skipping_mpp_boundaries(
    self,
    circuit: 'stim.Circuit',
    *,
    immune_qubit_indices: 'Set[int] | None' = None,
    immune_qubit_coords: 'Iterable[complex | float | int | Iterable[float | int]] | None' = None,
) -> 'stim.Circuit':
    """Adds noise to the circuit except for MPP operations at the start/end.

    Divides the circuit into three parts: mpp_start, body, mpp_end. The mpp
    sections grow from the ends of the circuit until they hit an instruction
    that's not an annotation or an MPP. Then body is the remaining circuit
    between the two ends. Noise is added to the body, and then the pieces
    are reassembled.
    """
```

<a name="gen.NoiseModel.si1000"></a>
```python
# gen.NoiseModel.si1000

# (in class gen.NoiseModel)
def si1000(
    p: 'float',
) -> 'NoiseModel':
    """Superconducting inspired noise.

    As defined in "A Fault-Tolerant Honeycomb Memory" https://arxiv.org/abs/2108.10457

    Small tweak when measurements aren't immediately followed by a reset: the measurement result
    is probabilistically flipped instead of the input qubit. The input qubit is depolarized
    after the measurement.
    """
```

<a name="gen.NoiseModel.uniform_depolarizing"></a>
```python
# gen.NoiseModel.uniform_depolarizing

# (in class gen.NoiseModel)
def uniform_depolarizing(
    p: 'float',
    *,
    single_qubit_only: 'bool' = False,
) -> 'NoiseModel':
    """Near-standard circuit depolarizing noise.

    Everything has the same parameter p.
    Single qubit clifford gates get single qubit depolarization.
    Two qubit clifford gates get single qubit depolarization.
    Dissipative gates have their result probabilistically bit flipped (or phase flipped if
    appropriate).

    Non-demolition measurement is treated a bit unusually in that it is the result that is
    flipped instead of the input qubit. The input qubit is depolarized.
    """
```

<a name="gen.NoiseRule"></a>
```python
# gen.NoiseRule

# (at top-level in the gen module)
class NoiseRule:
    """Describes how to add noise to an operation.
    """
```

<a name="gen.NoiseRule.__init__"></a>
```python
# gen.NoiseRule.__init__

# (in class gen.NoiseRule)
def __init__(
    self,
    *,
    before: 'dict[str, float | tuple[float, ...]] | None' = None,
    after: 'dict[str, float | tuple[float, ...]] | None' = None,
    flip_result: 'float' = 0,
):
    """

    Args:
        after: A dictionary mapping noise rule names to their probability argument.
            For example, {"DEPOLARIZE2": 0.01, "X_ERROR": 0.02} will add two qubit
            depolarization with parameter 0.01 and also add 2% bit flip noise. These
            noise channels occur after all other operations in the moment and are applied
            to the same targets as the relevant operation.
        flip_result: The probability that a measurement result should be reported incorrectly.
            Only valid when applied to operations that produce measurement results.
    """
```

<a name="gen.NoiseRule.append_noisy_version_of"></a>
```python
# gen.NoiseRule.append_noisy_version_of

# (in class gen.NoiseRule)
def append_noisy_version_of(
    self,
    *,
    split_op: 'stim.CircuitInstruction',
    out_during_moment: 'stim.Circuit',
    before_moments: 'collections.defaultdict[Any, stim.Circuit]',
    after_moments: 'collections.defaultdict[Any, stim.Circuit]',
    immune_qubit_indices: 'Set[int]',
) -> 'None':
```

<a name="gen.Patch"></a>
```python
# gen.Patch

# (at top-level in the gen module)
class Patch:
    """A collection of annotated stabilizers.
    """
```

<a name="gen.Patch.data_set"></a>
```python
# gen.Patch.data_set

# (in class gen.Patch)
class data_set:
```

<a name="gen.Patch.m2tile"></a>
```python
# gen.Patch.m2tile

# (in class gen.Patch)
class m2tile:
```

<a name="gen.Patch.measure_set"></a>
```python
# gen.Patch.measure_set

# (in class gen.Patch)
class measure_set:
```

<a name="gen.Patch.partitioned_tiles"></a>
```python
# gen.Patch.partitioned_tiles

# (in class gen.Patch)
class partitioned_tiles:
    """Returns the tiles of the patch, but split into non-overlapping groups.
    """
```

<a name="gen.Patch.to_svg"></a>
```python
# gen.Patch.to_svg

# (in class gen.Patch)
def to_svg(
    self,
    *,
    title: 'str | list[str] | None' = None,
    other: 'Patch | StabilizerCode | Iterable[Patch | StabilizerCode]' = (),
    show_order: 'bool' = False,
    show_measure_qubits: 'bool' = False,
    show_data_qubits: 'bool' = True,
    system_qubits: 'Iterable[complex]' = (),
    show_coords: 'bool' = True,
    opacity: 'float' = 1,
    show_obs: 'bool' = False,
    rows: 'int | None' = None,
    cols: 'int | None' = None,
    tile_color_func: 'Callable[[Tile], str] | None' = None,
) -> 'str_svg':
```

<a name="gen.Patch.used_set"></a>
```python
# gen.Patch.used_set

# (in class gen.Patch)
class used_set:
```

<a name="gen.Patch.with_edits"></a>
```python
# gen.Patch.with_edits

# (in class gen.Patch)
def with_edits(
    self,
    *,
    tiles: 'Iterable[Tile] | None' = None,
) -> 'Patch':
```

<a name="gen.Patch.with_only_x_tiles"></a>
```python
# gen.Patch.with_only_x_tiles

# (in class gen.Patch)
def with_only_x_tiles(
    self,
) -> 'Patch':
```

<a name="gen.Patch.with_only_y_tiles"></a>
```python
# gen.Patch.with_only_y_tiles

# (in class gen.Patch)
def with_only_y_tiles(
    self,
) -> 'Patch':
```

<a name="gen.Patch.with_only_z_tiles"></a>
```python
# gen.Patch.with_only_z_tiles

# (in class gen.Patch)
def with_only_z_tiles(
    self,
) -> 'Patch':
```

<a name="gen.Patch.with_remaining_degrees_of_freedom_as_logicals"></a>
```python
# gen.Patch.with_remaining_degrees_of_freedom_as_logicals

# (in class gen.Patch)
def with_remaining_degrees_of_freedom_as_logicals(
    self,
) -> 'StabilizerCode':
    """Solves for the logical observables, given only the stabilizers.
    """
```

<a name="gen.Patch.with_transformed_bases"></a>
```python
# gen.Patch.with_transformed_bases

# (in class gen.Patch)
def with_transformed_bases(
    self,
    basis_transform: "Callable[[Literal['X', 'Y', 'Z']], Literal['X', 'Y', 'Z']]",
) -> 'Patch':
```

<a name="gen.Patch.with_transformed_coords"></a>
```python
# gen.Patch.with_transformed_coords

# (in class gen.Patch)
def with_transformed_coords(
    self,
    coord_transform: 'Callable[[complex], complex]',
) -> 'Patch':
```

<a name="gen.Patch.with_xz_flipped"></a>
```python
# gen.Patch.with_xz_flipped

# (in class gen.Patch)
def with_xz_flipped(
    self,
) -> 'Patch':
```

<a name="gen.PauliMap"></a>
```python
# gen.PauliMap

# (at top-level in the gen module)
class PauliMap:
    """A qubit-to-pauli mapping.
    """
```

<a name="gen.PauliMap.__init__"></a>
```python
# gen.PauliMap.__init__

# (in class gen.PauliMap)
def __init__(
    self,
    mapping: "dict[complex, Literal['X', 'Y', 'Z'] | str] | dict[Literal['X', 'Y', 'Z'] | str, complex | Iterable[complex]] | PauliMap | Tile | stim.PauliString | None" = None,
    *,
    name: 'Any' = None,
):
    """Initializes a PauliMap using maps of Paulis to/from qubits.

    Args:
        mapping: The association between qubits and paulis, specifiable in a variety of ways.
        name: Defaults to None (no name). Can be set to an arbitrary hashable equatable value,
            in order to identify the Pauli map. A common convention used in the library is that
            named Pauli maps correspond to logical operators.
    """
```

<a name="gen.PauliMap.anticommutes"></a>
```python
# gen.PauliMap.anticommutes

# (in class gen.PauliMap)
def anticommutes(
    self,
    other: 'PauliMap',
) -> 'bool':
    """Determines if the pauli map anticommutes with another pauli map.
    """
```

<a name="gen.PauliMap.commutes"></a>
```python
# gen.PauliMap.commutes

# (in class gen.PauliMap)
def commutes(
    self,
    other: 'PauliMap',
) -> 'bool':
    """Determines if the pauli map commutes with another pauli map.
    """
```

<a name="gen.PauliMap.from_xs"></a>
```python
# gen.PauliMap.from_xs

# (in class gen.PauliMap)
def from_xs(
    xs: 'Iterable[complex]',
    *,
    name: 'Any' = None,
) -> 'PauliMap':
    """Returns a PauliMap mapping the given qubits to the X basis.
    """
```

<a name="gen.PauliMap.from_ys"></a>
```python
# gen.PauliMap.from_ys

# (in class gen.PauliMap)
def from_ys(
    ys: 'Iterable[complex]',
    *,
    name: 'Any' = None,
) -> 'PauliMap':
    """Returns a PauliMap mapping the given qubits to the Y basis.
    """
```

<a name="gen.PauliMap.from_zs"></a>
```python
# gen.PauliMap.from_zs

# (in class gen.PauliMap)
def from_zs(
    zs: 'Iterable[complex]',
    *,
    name: 'Any' = None,
) -> 'PauliMap':
    """Returns a PauliMap mapping the given qubits to the Z basis.
    """
```

<a name="gen.PauliMap.get"></a>
```python
# gen.PauliMap.get

# (in class gen.PauliMap)
def get(
    self,
    key: 'complex',
    default: 'Any' = None,
) -> 'Any':
```

<a name="gen.PauliMap.items"></a>
```python
# gen.PauliMap.items

# (in class gen.PauliMap)
def items(
    self,
) -> "Iterable[tuple[complex, Literal['X', 'Y', 'Z']]]":
    """Returns the (qubit, basis) pairs of the PauliMap.
    """
```

<a name="gen.PauliMap.keys"></a>
```python
# gen.PauliMap.keys

# (in class gen.PauliMap)
def keys(
    self,
) -> 'Set[complex]':
    """Returns the qubits of the PauliMap.
    """
```

<a name="gen.PauliMap.to_stim_pauli_string"></a>
```python
# gen.PauliMap.to_stim_pauli_string

# (in class gen.PauliMap)
def to_stim_pauli_string(
    self,
    q2i: 'dict[complex, int]',
    *,
    num_qubits: 'int | None' = None,
) -> 'stim.PauliString':
    """Converts into a stim.PauliString.
    """
```

<a name="gen.PauliMap.to_stim_targets"></a>
```python
# gen.PauliMap.to_stim_targets

# (in class gen.PauliMap)
def to_stim_targets(
    self,
    q2i: 'dict[complex, int]',
) -> 'list[stim.GateTarget]':
    """Converts into a stim combined pauli target like 'X1*Y2*Z3'.
    """
```

<a name="gen.PauliMap.to_tile"></a>
```python
# gen.PauliMap.to_tile

# (in class gen.PauliMap)
def to_tile(
    self,
) -> 'Tile':
    """Converts the PauliMap into a gen.Tile.
    """
```

<a name="gen.PauliMap.values"></a>
```python
# gen.PauliMap.values

# (in class gen.PauliMap)
def values(
    self,
) -> "Iterable[Literal['X', 'Y', 'Z']]":
    """Returns the bases used by the PauliMap.
    """
```

<a name="gen.PauliMap.with_basis"></a>
```python
# gen.PauliMap.with_basis

# (in class gen.PauliMap)
def with_basis(
    self,
    basis: "Literal['X', 'Y', 'Z']",
) -> 'PauliMap':
    """Returns the same PauliMap, but with all its qubits mapped to the given basis.
    """
```

<a name="gen.PauliMap.with_name"></a>
```python
# gen.PauliMap.with_name

# (in class gen.PauliMap)
def with_name(
    self,
    name: 'Any',
) -> 'PauliMap':
    """Returns the same PauliMap, but with the given name.

    Names are used to identify logical operators.
    """
```

<a name="gen.PauliMap.with_transformed_coords"></a>
```python
# gen.PauliMap.with_transformed_coords

# (in class gen.PauliMap)
def with_transformed_coords(
    self,
    transform: 'Callable[[complex], complex]',
) -> 'PauliMap':
    """Returns the same PauliMap but with coordinates transformed by the given function.
    """
```

<a name="gen.PauliMap.with_xy_flipped"></a>
```python
# gen.PauliMap.with_xy_flipped

# (in class gen.PauliMap)
def with_xy_flipped(
    self,
) -> 'PauliMap':
    """Returns the same PauliMap, but with all qubits conjugated by H_XY.
    """
```

<a name="gen.PauliMap.with_xz_flipped"></a>
```python
# gen.PauliMap.with_xz_flipped

# (in class gen.PauliMap)
def with_xz_flipped(
    self,
) -> 'PauliMap':
    """Returns the same PauliMap, but with all qubits conjugated by H.
    """
```

<a name="gen.ResetLayer"></a>
```python
# gen.ResetLayer

# (at top-level in the gen module)
@dataclasses.dataclass
class ResetLayer:
    """A layer of reset gates.
    """
    targets: dict[int, Literal['X', 'Y', 'Z']]
```

<a name="gen.ResetLayer.append_into_stim_circuit"></a>
```python
# gen.ResetLayer.append_into_stim_circuit

# (in class gen.ResetLayer)
def append_into_stim_circuit(
    self,
    out: 'stim.Circuit',
) -> 'None':
```

<a name="gen.ResetLayer.copy"></a>
```python
# gen.ResetLayer.copy

# (in class gen.ResetLayer)
def copy(
    self,
) -> 'ResetLayer':
```

<a name="gen.ResetLayer.locally_optimized"></a>
```python
# gen.ResetLayer.locally_optimized

# (in class gen.ResetLayer)
def locally_optimized(
    self,
    next_layer: 'Layer | None',
) -> 'list[Layer | None]':
```

<a name="gen.ResetLayer.to_z_basis"></a>
```python
# gen.ResetLayer.to_z_basis

# (in class gen.ResetLayer)
def to_z_basis(
    self,
) -> 'list[Layer]':
```

<a name="gen.ResetLayer.touched"></a>
```python
# gen.ResetLayer.touched

# (in class gen.ResetLayer)
def touched(
    self,
) -> 'set[int]':
```

<a name="gen.RotationLayer"></a>
```python
# gen.RotationLayer

# (at top-level in the gen module)
@dataclasses.dataclass
class RotationLayer:
    """A layer of single qubit Clifford rotation gates.
    """
    named_rotations: dict[int, str]
```

<a name="gen.RotationLayer.append_into_stim_circuit"></a>
```python
# gen.RotationLayer.append_into_stim_circuit

# (in class gen.RotationLayer)
def append_into_stim_circuit(
    self,
    out: 'stim.Circuit',
) -> 'None':
```

<a name="gen.RotationLayer.append_named_rotation"></a>
```python
# gen.RotationLayer.append_named_rotation

# (in class gen.RotationLayer)
def append_named_rotation(
    self,
    name: 'str',
    target: 'int',
):
```

<a name="gen.RotationLayer.copy"></a>
```python
# gen.RotationLayer.copy

# (in class gen.RotationLayer)
def copy(
    self,
) -> 'RotationLayer':
```

<a name="gen.RotationLayer.inverse"></a>
```python
# gen.RotationLayer.inverse

# (in class gen.RotationLayer)
def inverse(
    self,
) -> 'RotationLayer':
```

<a name="gen.RotationLayer.is_vacuous"></a>
```python
# gen.RotationLayer.is_vacuous

# (in class gen.RotationLayer)
def is_vacuous(
    self,
) -> 'bool':
```

<a name="gen.RotationLayer.locally_optimized"></a>
```python
# gen.RotationLayer.locally_optimized

# (in class gen.RotationLayer)
def locally_optimized(
    self,
    next_layer: 'Layer | None',
) -> 'list[Layer | None]':
```

<a name="gen.RotationLayer.prepend_named_rotation"></a>
```python
# gen.RotationLayer.prepend_named_rotation

# (in class gen.RotationLayer)
def prepend_named_rotation(
    self,
    name: 'str',
    target: 'int',
):
```

<a name="gen.RotationLayer.touched"></a>
```python
# gen.RotationLayer.touched

# (in class gen.RotationLayer)
def touched(
    self,
) -> 'set[int]':
```

<a name="gen.StabilizerCode"></a>
```python
# gen.StabilizerCode

# (at top-level in the gen module)
class StabilizerCode:
    """This class stores the stabilizers and logicals of a stabilizer code.

    The exact semantics of the class are somewhat loose. For example, by default
    this class doesn't verify that its fields actually form a valid stabilizer
    code. This is so that the class can be used as a sort of useful data dumping
    ground even in cases where what is being built isn't a stabilizer code. For
    example, you can store a gauge code in the fields... it's just that methods
    like 'make_code_capacity_circuit' will no longer work.

    The stabilizers are stored as a `gen.Patch`. A patch is like a list of `gen.PauliMap`,
    except it actually stores `gen.Tile` instances so additional annotations can be added
    and additional utility methods are easily available.
    """
```

<a name="gen.StabilizerCode.__init__"></a>
```python
# gen.StabilizerCode.__init__

# (in class gen.StabilizerCode)
def __init__(
    self,
    stabilizers: 'Iterable[Tile | PauliMap] | Patch | None' = None,
    *,
    logicals: 'Iterable[PauliMap | tuple[PauliMap, PauliMap]]' = (),
    scattered_logicals: 'Iterable[PauliMap]' = (),
):
    """

    Args:
        stabilizers: The stabilizers of the code, specified as a Patch
        logicals: The logical qubits and/or observables of the code. Each entry should be
            either a pair of anti-commuting gen.PauliMap (e.g. the X and Z observables of the
            logical qubit) or a single gen.PauliMap (e.g. just the X observable).
        scattered_logicals: Logical operators with arbitrary commutation relationships to each
            other. Still expected to commute with the stabilizers.
    """
```

<a name="gen.StabilizerCode.as_interface"></a>
```python
# gen.StabilizerCode.as_interface

# (in class gen.StabilizerCode)
def as_interface(
    self,
) -> 'gen.ChunkInterface':
```

<a name="gen.StabilizerCode.concat_over"></a>
```python
# gen.StabilizerCode.concat_over

# (in class gen.StabilizerCode)
def concat_over(
    self,
    under: 'StabilizerCode',
    *,
    skip_inner_stabilizers: 'bool' = False,
) -> 'StabilizerCode':
    """Computes the interleaved concatenation of two stabilizer codes.
    """
```

<a name="gen.StabilizerCode.data_set"></a>
```python
# gen.StabilizerCode.data_set

# (in class gen.StabilizerCode)
class data_set:
```

<a name="gen.StabilizerCode.find_distance"></a>
```python
# gen.StabilizerCode.find_distance

# (in class gen.StabilizerCode)
def find_distance(
    self,
    *,
    max_search_weight: 'int',
) -> 'int':
```

<a name="gen.StabilizerCode.find_logical_error"></a>
```python
# gen.StabilizerCode.find_logical_error

# (in class gen.StabilizerCode)
def find_logical_error(
    self,
    *,
    max_search_weight: 'int',
) -> 'list[stim.ExplainedError]':
```

<a name="gen.StabilizerCode.flat_logicals"></a>
```python
# gen.StabilizerCode.flat_logicals

# (in class gen.StabilizerCode)
class flat_logicals:
    """Returns a list of the logical operators defined by the stabilizer code.

    It's "flat" because paired X/Z logicals are returned separately instead of
    as a tuple.
    """
```

<a name="gen.StabilizerCode.from_patch_with_inferred_observables"></a>
```python
# gen.StabilizerCode.from_patch_with_inferred_observables

# (in class gen.StabilizerCode)
def from_patch_with_inferred_observables(
    patch: 'Patch',
) -> 'StabilizerCode':
```

<a name="gen.StabilizerCode.get_observable_by_basis"></a>
```python
# gen.StabilizerCode.get_observable_by_basis

# (in class gen.StabilizerCode)
def get_observable_by_basis(
    self,
    index: 'int',
    basis: "Literal['X', 'Y', 'Z'] | str",
    *,
    default: 'Any' = '__!not_specified',
) -> 'PauliMap':
```

<a name="gen.StabilizerCode.list_pure_basis_observables"></a>
```python
# gen.StabilizerCode.list_pure_basis_observables

# (in class gen.StabilizerCode)
def list_pure_basis_observables(
    self,
    basis: "Literal['X', 'Y', 'Z']",
) -> 'list[PauliMap]':
```

<a name="gen.StabilizerCode.make_code_capacity_circuit"></a>
```python
# gen.StabilizerCode.make_code_capacity_circuit

# (in class gen.StabilizerCode)
def make_code_capacity_circuit(
    self,
    *,
    noise: 'float | NoiseRule',
    metadata_func: 'Callable[[gen.Flow], gen.FlowMetadata]' = lambda _: FlowMetadata(),
) -> 'stim.Circuit':
    """Produces a code capacity noisy memory experiment circuit for the stabilizer code.
    """
```

<a name="gen.StabilizerCode.make_phenom_circuit"></a>
```python
# gen.StabilizerCode.make_phenom_circuit

# (in class gen.StabilizerCode)
def make_phenom_circuit(
    self,
    *,
    noise: 'float | NoiseRule',
    rounds: 'int',
    metadata_func: 'Callable[[gen.Flow], gen.FlowMetadata]' = lambda _: FlowMetadata(),
) -> 'stim.Circuit':
    """Produces a phenomenological noise memory experiment circuit for the stabilizer code.
    """
```

<a name="gen.StabilizerCode.measure_set"></a>
```python
# gen.StabilizerCode.measure_set

# (in class gen.StabilizerCode)
class measure_set:
```

<a name="gen.StabilizerCode.patch"></a>
```python
# gen.StabilizerCode.patch

# (in class gen.StabilizerCode)
@property
def patch(
    self,
):
    """Returns the gen.Patch storing the stabilizers of the code.
    """
```

<a name="gen.StabilizerCode.physical_to_logical"></a>
```python
# gen.StabilizerCode.physical_to_logical

# (in class gen.StabilizerCode)
def physical_to_logical(
    self,
    ps: 'stim.PauliString',
) -> 'PauliMap':
    """Maps a physical qubit string into a logical qubit string.

    Requires that all logicals be specified as X/Z tuples.
    """
```

<a name="gen.StabilizerCode.tiles"></a>
```python
# gen.StabilizerCode.tiles

# (in class gen.StabilizerCode)
@property
def tiles(
    self,
):
    """Returns the tiles of the code's stabilizer patch.
    """
```

<a name="gen.StabilizerCode.to_svg"></a>
```python
# gen.StabilizerCode.to_svg

# (in class gen.StabilizerCode)
def to_svg(
    self,
    *,
    title: 'str | list[str] | None' = None,
    canvas_height: 'int | None' = None,
    show_order: 'bool' = False,
    show_measure_qubits: 'bool' = False,
    show_data_qubits: 'bool' = True,
    system_qubits: 'Iterable[complex]' = (),
    opacity: 'float' = 1,
    show_coords: 'bool' = True,
    show_obs: 'bool' = True,
    other: 'gen.StabilizerCode | Patch | Iterable[gen.StabilizerCode | Patch] | None' = None,
    tile_color_func: 'Callable[[gen.Tile], str | tuple[float, float, float] | tuple[float, float, float, float] | None] | None' = None,
    rows: 'int | None' = None,
    cols: 'int | None' = None,
    find_logical_err_max_weight: 'int | None' = None,
    stabilizer_style: "Literal['polygon', 'circles'] | None" = 'polygon',
    observable_style: "Literal['label', 'polygon', 'circles']" = 'label',
) -> 'str_svg':
    """Returns an SVG diagram of the stabilizer code.
    """
```

<a name="gen.StabilizerCode.transversal_init_chunk"></a>
```python
# gen.StabilizerCode.transversal_init_chunk

# (in class gen.StabilizerCode)
def transversal_init_chunk(
    self,
    *,
    basis: "Literal['X', 'Y', 'Z'] | str | gen.PauliMap | dict[complex, str | Literal['X', 'Y', 'Z']]",
) -> 'gen.Chunk':
    """Returns a chunk that describes initializing the stabilizer code with given reset bases.

    Stabilizers that anticommute with the resets will be discarded flows.

    The returned chunk isn't guaranteed to be fault tolerant.
    """
```

<a name="gen.StabilizerCode.transversal_measure_chunk"></a>
```python
# gen.StabilizerCode.transversal_measure_chunk

# (in class gen.StabilizerCode)
def transversal_measure_chunk(
    self,
    *,
    basis: "Literal['X', 'Y', 'Z'] | str | gen.PauliMap | dict[complex, str | Literal['X', 'Y', 'Z']]",
) -> 'gen.Chunk':
    """Returns a chunk that describes measuring the stabilizer code with given measure bases.

    Stabilizers that anticommute with the measurements will be discarded flows.

    The returned chunk isn't guaranteed to be fault tolerant.
    """
```

<a name="gen.StabilizerCode.used_set"></a>
```python
# gen.StabilizerCode.used_set

# (in class gen.StabilizerCode)
class used_set:
```

<a name="gen.StabilizerCode.verify"></a>
```python
# gen.StabilizerCode.verify

# (in class gen.StabilizerCode)
def verify(
    self,
) -> 'None':
    """Verifies observables and stabilizers relate as a stabilizer code.

    All stabilizers should commute with each other.
    All stabilizers should commute with all observables.
    Same-index X and Z observables should anti-commute.
    All other observable pairs should commute.
    """
```

<a name="gen.StabilizerCode.verify_distance_is_at_least_2"></a>
```python
# gen.StabilizerCode.verify_distance_is_at_least_2

# (in class gen.StabilizerCode)
def verify_distance_is_at_least_2(
    self,
):
    """Verifies undetected logical errors require at least 2 physical errors.

    Verifies using a code capacity noise model.
    """
```

<a name="gen.StabilizerCode.verify_distance_is_at_least_3"></a>
```python
# gen.StabilizerCode.verify_distance_is_at_least_3

# (in class gen.StabilizerCode)
def verify_distance_is_at_least_3(
    self,
):
    """Verifies undetected logical errors require at least 3 physical errors.

    Verifies using a code capacity noise model.
    """
```

<a name="gen.StabilizerCode.with_edits"></a>
```python
# gen.StabilizerCode.with_edits

# (in class gen.StabilizerCode)
def with_edits(
    self,
    *,
    stabilizers: 'Iterable[Tile | PauliMap] | Patch | None' = None,
    logicals: 'Iterable[PauliMap | tuple[PauliMap, PauliMap]] | None' = None,
) -> 'StabilizerCode':
```

<a name="gen.StabilizerCode.with_integer_coordinates"></a>
```python
# gen.StabilizerCode.with_integer_coordinates

# (in class gen.StabilizerCode)
def with_integer_coordinates(
    self,
) -> 'StabilizerCode':
    """Returns an equivalent stabilizer code, but with all qubit on Gaussian integers.
    """
```

<a name="gen.StabilizerCode.with_observables_from_basis"></a>
```python
# gen.StabilizerCode.with_observables_from_basis

# (in class gen.StabilizerCode)
def with_observables_from_basis(
    self,
    basis: "Literal['X', 'Y', 'Z']",
) -> 'StabilizerCode':
```

<a name="gen.StabilizerCode.with_remaining_degrees_of_freedom_as_logicals"></a>
```python
# gen.StabilizerCode.with_remaining_degrees_of_freedom_as_logicals

# (in class gen.StabilizerCode)
def with_remaining_degrees_of_freedom_as_logicals(
    self,
) -> 'StabilizerCode':
    """Solves for the logical observables, given only the stabilizers.
    """
```

<a name="gen.StabilizerCode.with_transformed_coords"></a>
```python
# gen.StabilizerCode.with_transformed_coords

# (in class gen.StabilizerCode)
def with_transformed_coords(
    self,
    coord_transform: 'Callable[[complex], complex]',
) -> 'StabilizerCode':
    """Returns the same stabilizer code, but with coordinates transformed by the given function.
    """
```

<a name="gen.StabilizerCode.with_xz_flipped"></a>
```python
# gen.StabilizerCode.with_xz_flipped

# (in class gen.StabilizerCode)
def with_xz_flipped(
    self,
) -> 'StabilizerCode':
    """Returns the same stabilizer code, but with all qubits Hadamard conjugated.
    """
```

<a name="gen.StabilizerCode.x_basis_subset"></a>
```python
# gen.StabilizerCode.x_basis_subset

# (in class gen.StabilizerCode)
def x_basis_subset(
    self,
) -> 'StabilizerCode':
```

<a name="gen.StabilizerCode.z_basis_subset"></a>
```python
# gen.StabilizerCode.z_basis_subset

# (in class gen.StabilizerCode)
def z_basis_subset(
    self,
) -> 'StabilizerCode':
```

<a name="gen.StimCircuitLoom"></a>
```python
# gen.StimCircuitLoom

# (at top-level in the gen module)
class StimCircuitLoom:
    """class supporting the combining of stim circuits in space.

    for standard usage, call StimCircuitLoom.weave(...), which returns the weaved circuit
    for usage details, see the docstring to that function

    for complex usage, you can instantiate a loom StimCircuitLoom(...)
    This is lets you access details of the weaving afterwards, such as the measurement mapping
    """
```

<a name="gen.StimCircuitLoom.weave"></a>
```python
# gen.StimCircuitLoom.weave

# (in class gen.StimCircuitLoom)
class weave:
    """Combines two stim circuits instruction by instruction.

    Example usage:
        StimCircuitLoom.weave(circuit_0, circuit_1) -> stim.Circuit

    Expects that the input circuit have 'matching instructions', in that they
    contain exactly the same sequence of instructions which can be matched up
    1-to-1. This may require one circuit to have instructions with no targets,
    purely to match instructions in the other circuit. Exceptions to this are
    the annotation instructions DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS,
    and SHIFT_COORDS, which do not need a matching statement in the other
    circuit. This may not be what you want, as it will produce duplicate
    DETECTOR or QUBIT_COORD instructions if they are included in both circuits.
    The annotation TICK is considered a matching instruction.

    Generally, instructions are combined by placing all targets from the
    first circuit instruction, followed by all targets from the second.

    In most gates, if a gate target is present in the first instruction
    target list, it is removed from the second instructions target list.
    As such, we do not permit instructions in the input circuits to have
    duplicate targets. This avoids the ambiguity of deciding whether one
    or both duplicates between circuits have to match up.

    Measure record targets are adjusted to point to the correct record in the
    combined circuit e.g. DETECTOR rec[-1] or CX rec[-1] 1

    Sweep bits are not handled by default, and will produce a ValueError.
    If sweep_bit_func is provided, it will be used to produce new sweep bit
    targets as follows:
        new_sweep_bit_index = sweep_bit_func(circuit_index, sweep_bit_index)
        where:
            circuit_index = 0 for circuit_0 and 1 for circuit_1
            sweep_bit_index is the sweep bit index used in the input circuit
    """
```

<a name="gen.StimCircuitLoom.weaved_target_rec_from_c0"></a>
```python
# gen.StimCircuitLoom.weaved_target_rec_from_c0

# (in class gen.StimCircuitLoom)
def weaved_target_rec_from_c0(
    self,
    target_rec: int,
) -> int:
    """given a target rec in circuit_0, return the equiv rec in the weaved circuit.

    args:
        target_rec: a valid measurement record target in the input circuit
            follows python indexing semantics:
            can be either positive (counting from the start of the circuit, 0 indexed)
            or negative (counting from the end backwards, last measurement is  [-1])
            The second is compatible with stim instruction target rec values

    returns:
        The same measurements target rec in the weaved circuit.
            Always returns a negative 'lookback' compatible with a stim circuit
            Add StimCircuitWeave.circuit.num_measurements for an absolute measurement index
    """
```

<a name="gen.StimCircuitLoom.weaved_target_rec_from_c1"></a>
```python
# gen.StimCircuitLoom.weaved_target_rec_from_c1

# (in class gen.StimCircuitLoom)
def weaved_target_rec_from_c1(
    self,
    target_rec: int,
) -> int:
    """given a target rec in circuit_1, return the equiv rec in the weaved circuit.
    """
```

<a name="gen.TextData"></a>
```python
# gen.TextData

# (at top-level in the gen module)
class TextData:
```

<a name="gen.Tile"></a>
```python
# gen.Tile

# (at top-level in the gen module)
class Tile:
    """A stabilizer with some associated metadata.

    The exact meaning of the tile's fields are often context dependent. For example,
    different circuits will use the measure qubit in different ways (or not at all)
    and the flags could be essentially anything at all. Tile is intended to be useful
    as an intermediate step in the production of a circuit.

    For example, it's much easier to create a color code circuit when you have a list
    of the hexagonal and trapezoidal shapes making up the color code. So it's natural to
    split the color code circuit generation problem into two steps: (1) making the shapes
    then (2) making the circuit given the shapes. In other words, deal with the spatial
    complexities first then deal with the temporal complexities second. The Tile class
    is a reasonable representation for the shapes, because:

    - The X/Z basis of the stabilizer can be stored in the `bases` field.
    - The red/green/blue coloring can be stored as flags.
    - The ancilla qubits for the shapes be stored as measure_qubit values.
    - You can get diagrams of the shapes by passing the tiles into a `gen.Patch`.
    - You can verify the tiles form a code by passing the patch into a `gen.StabilizerCode`.
    """
```

<a name="gen.Tile.__init__"></a>
```python
# gen.Tile.__init__

# (in class gen.Tile)
def __init__(
    self,
    *,
    bases: 'str',
    data_qubits: 'Iterable[complex | None]',
    measure_qubit: 'complex | None' = None,
    flags: 'Iterable[str]' = (),
):
    """

    Args:
        bases: Basis of the stabilizer. A string of XYZ characters the same
            length as the data_qubits argument. It is permitted to
            give a single-character string, which will automatically be
            expanded to the full length. For example, 'X' will become 'XXXX'
            if there are four data qubits.
        measure_qubit: The ancilla qubit used to measure the stabilizer.
        data_qubits: The data qubits in the stabilizer, in the order
            that they are interacted with. Some entries may be None,
            indicating that no data qubit is interacted with during the
            corresponding interaction layer.
    """
```

<a name="gen.Tile.basis"></a>
```python
# gen.Tile.basis

# (in class gen.Tile)
class basis:
```

<a name="gen.Tile.center"></a>
```python
# gen.Tile.center

# (in class gen.Tile)
def center(
    self,
) -> 'complex':
```

<a name="gen.Tile.data_set"></a>
```python
# gen.Tile.data_set

# (in class gen.Tile)
class data_set:
```

<a name="gen.Tile.to_pauli_map"></a>
```python
# gen.Tile.to_pauli_map

# (in class gen.Tile)
def to_pauli_map(
    self,
) -> 'PauliMap':
```

<a name="gen.Tile.used_set"></a>
```python
# gen.Tile.used_set

# (in class gen.Tile)
class used_set:
```

<a name="gen.Tile.with_bases"></a>
```python
# gen.Tile.with_bases

# (in class gen.Tile)
def with_bases(
    self,
    bases: 'str',
) -> 'Tile':
```

<a name="gen.Tile.with_basis"></a>
```python
# gen.Tile.with_basis

# (in class gen.Tile)
def with_basis(
    self,
    bases: 'str',
) -> 'Tile':
```

<a name="gen.Tile.with_data_qubit_cleared"></a>
```python
# gen.Tile.with_data_qubit_cleared

# (in class gen.Tile)
def with_data_qubit_cleared(
    self,
    q: 'complex',
) -> 'Tile':
```

<a name="gen.Tile.with_edits"></a>
```python
# gen.Tile.with_edits

# (in class gen.Tile)
def with_edits(
    self,
    *,
    bases: 'str | None' = None,
    measure_qubit: "complex | None | Literal['unspecified']" = 'unspecified',
    data_qubits: 'Iterable[complex | None] | None' = None,
    flags: 'Iterable[str] | None' = None,
) -> 'Tile':
```

<a name="gen.Tile.with_transformed_bases"></a>
```python
# gen.Tile.with_transformed_bases

# (in class gen.Tile)
def with_transformed_bases(
    self,
    basis_transform: "Callable[[Literal['X', 'Y', 'Z']], Literal['X', 'Y', 'Z']]",
) -> 'Tile':
```

<a name="gen.Tile.with_transformed_coords"></a>
```python
# gen.Tile.with_transformed_coords

# (in class gen.Tile)
def with_transformed_coords(
    self,
    coord_transform: 'Callable[[complex], complex]',
) -> 'Tile':
```

<a name="gen.Tile.with_xz_flipped"></a>
```python
# gen.Tile.with_xz_flipped

# (in class gen.Tile)
def with_xz_flipped(
    self,
) -> 'Tile':
```

<a name="gen.append_reindexed_content_to_circuit"></a>
```python
# gen.append_reindexed_content_to_circuit

# (at top-level in the gen module)
def append_reindexed_content_to_circuit(
    *,
    out_circuit: 'stim.Circuit',
    content: 'stim.Circuit',
    qubit_i2i: 'dict[int, int]',
    obs_i2i: "dict[int, int | Literal['discard']]",
    rewrite_detector_time_coordinates: 'bool' = False,
) -> 'None':
    """Reindexes content and appends it to a circuit.

    Note that QUBIT_COORDS instructions are skipped.

    Args:
        out_circuit: The output circuit. The circuit being edited.
        content: The circuit to be appended to the output circuit.
        qubit_i2i: A dictionary specifying how qubit indices are remapped. Indices outside the
            map are not changed.
        obs_i2i: A dictionary specifying how observable indices are remapped. Indices outside the
            map are not changed.
        rewrite_detector_time_coordinates: Defaults to False. When set to True, SHIFT_COORD and
            DETECTOR instructions are automatically rewritten to track the passage of time without
            using the same detector position twice at the same time.
    """
```

<a name="gen.circuit_to_cycle_code_slices"></a>
```python
# gen.circuit_to_cycle_code_slices

# (at top-level in the gen module)
def circuit_to_cycle_code_slices(
    circuit: 'stim.Circuit',
) -> 'dict[int, StabilizerCode]':
```

<a name="gen.circuit_to_dem_target_measurement_records_map"></a>
```python
# gen.circuit_to_dem_target_measurement_records_map

# (at top-level in the gen module)
def circuit_to_dem_target_measurement_records_map(
    circuit: 'stim.Circuit',
) -> 'dict[stim.DemTarget, list[int]]':
```

<a name="gen.circuit_with_xz_flipped"></a>
```python
# gen.circuit_with_xz_flipped

# (at top-level in the gen module)
def circuit_with_xz_flipped(
    circuit: 'stim.Circuit',
) -> 'stim.Circuit':
```

<a name="gen.compile_chunks_into_circuit"></a>
```python
# gen.compile_chunks_into_circuit

# (at top-level in the gen module)
def compile_chunks_into_circuit(
    chunks: 'list[Chunk | ChunkLoop | ChunkReflow]',
    *,
    use_magic_time_boundaries: 'bool' = False,
    metadata_func: 'Callable[[Flow], FlowMetadata]' = lambda _: FlowMetadata(),
) -> 'stim.Circuit':
    """Stitches together a series of chunks into a fault tolerant circuit.

    Args:
        chunks: The sequence of chunks to compile into a circuit.
        use_magic_time_boundaries: Defaults to False. When False, an error will be raised if the
            first chunk has any non-empty input flows or the last chunk has any non-empty output
            flows (indicating the circuit is not complete). When True, the compiler will
            automatically close those flows by inserting MPP and OBSERVABLE_INCLUDE instructions to
            explain the dangling flows.
        metadata_func: Defaults to using no metadata. This function should take a gen.Flow and
            return a gen.FlowMetadata. The metadata is used for adding tags to coordinates to
            DETECTOR instructions and tags to DETECTOR/OBSERVABLE_INCLUDE instructions.

    Returns:
        The compiled circuit.
    """
```

<a name="gen.complex_key"></a>
```python
# gen.complex_key

# (at top-level in the gen module)
def complex_key(
    c: 'complex',
) -> 'Any':
```

<a name="gen.count_measurement_layers"></a>
```python
# gen.count_measurement_layers

# (at top-level in the gen module)
def count_measurement_layers(
    circuit: 'stim.Circuit',
) -> 'int':
```

<a name="gen.find_d1_error"></a>
```python
# gen.find_d1_error

# (at top-level in the gen module)
def find_d1_error(
    obj: 'stim.Circuit | stim.DetectorErrorModel',
) -> 'stim.ExplainedError | stim.DemInstruction | None':
```

<a name="gen.find_d2_error"></a>
```python
# gen.find_d2_error

# (at top-level in the gen module)
def find_d2_error(
    obj: 'stim.Circuit | stim.DetectorErrorModel',
) -> 'list[stim.ExplainedError] | stim.DetectorErrorModel | None':
```

<a name="gen.gate_counts_for_circuit"></a>
```python
# gen.gate_counts_for_circuit

# (at top-level in the gen module)
def gate_counts_for_circuit(
    circuit: 'stim.Circuit',
) -> 'collections.Counter[str]':
    """Determines gates used by a circuit, disambiguating MPP/feedback cases.

    MPP instructions are expanded into what they actually measure, such as
    "MXX" for MPP X1*X2 and "MXYZ" for MPP X4*Y5*Z7.

    Feedback instructions like `CX rec[-1] 0` become the gate "feedback".

    Sweep instructions like `CX sweep[2] 0` become the gate "sweep".
    """
```

<a name="gen.gates_used_by_circuit"></a>
```python
# gen.gates_used_by_circuit

# (at top-level in the gen module)
def gates_used_by_circuit(
    circuit: 'stim.Circuit',
) -> 'set[str]':
    """Determines gates used by a circuit, disambiguating MPP/feedback cases.

    MPP instructions are expanded into what they actually measure, such as
    "MXX" for MPP X1*X2 and "MXYZ" for MPP X4*Y5*Z7.

    Feedback instructions like `CX rec[-1] 0` become the gate "feedback".

    Sweep instructions like `CX sweep[2] 0` become the gate "sweep".
    """
```

<a name="gen.gltf_model"></a>
```python
# gen.gltf_model

# (at top-level in the gen module)
@dataclasses.dataclass
class gltf_model:
    """A pygltflib.GLTF2 augmented with _repr_html_ and a `write_to` method.
    """
    extensions: Optional[Dict[str, Any]]
    extras: Optional[Dict[str, Any]]
    accessors: List[pygltflib.Accessor]
    animations: List[pygltflib.Animation]
    asset: <class 'pygltflib.Asset'>
    bufferViews: List[pygltflib.BufferView]
    buffers: List[pygltflib.Buffer]
    cameras: List[pygltflib.Camera]
    extensionsUsed: List[str]
    extensionsRequired: List[str]
    images: List[pygltflib.Image]
    materials: List[pygltflib.Material]
    meshes: List[pygltflib.Mesh]
    nodes: List[pygltflib.Node]
    samplers: List[pygltflib.Sampler]
    scene: <class 'int'> = None
    scenes: List[pygltflib.Scene]
    skins: List[pygltflib.Skin]
    textures: List[pygltflib.Texture]
```

<a name="gen.gltf_model.write_viewer_to"></a>
```python
# gen.gltf_model.write_viewer_to

# (in class gen.gltf_model)
def write_viewer_to(
    self,
    path: 'str | pathlib.Path | io.IOBase',
):
```

<a name="gen.gltf_model_from_colored_triangle_data"></a>
```python
# gen.gltf_model_from_colored_triangle_data

# (at top-level in the gen module)
def gltf_model_from_colored_triangle_data(
    colored_triangle_data: 'list[ColoredTriangleData]',
    *,
    colored_line_data: 'Sequence[ColoredLineData]' = (),
    text_data: 'Sequence[TextData]' = (),
) -> 'gltf_model':
```

<a name="gen.min_max_complex"></a>
```python
# gen.min_max_complex

# (at top-level in the gen module)
def min_max_complex(
    coords: 'Iterable[complex]',
    *,
    default: 'complex | None' = None,
) -> 'tuple[complex, complex]':
    """Computes the bounding box of a collection of complex numbers.

    Args:
        coords: The complex numbers to place a bounding box around.
        default: If no elements are included, the bounding box will cover this
            single value when the collection of complex numbers is empty. If
            this argument isn't set (or is set to None), an exception will be
            raised instead when given an empty collection.

    Returns:
        A pair of complex values (c_min, c_max) where c_min's real component
        where c_min is the minimum corner of the bounding box and c_max is the
        maximum corner of the bounding box.
    """
```

<a name="gen.sorted_complex"></a>
```python
# gen.sorted_complex

# (at top-level in the gen module)
def sorted_complex(
    values: 'Iterable[complex]',
) -> 'list[complex]':
```

<a name="gen.stim_circuit_html_viewer"></a>
```python
# gen.stim_circuit_html_viewer

# (at top-level in the gen module)
def stim_circuit_html_viewer(
    circuit: 'stim.Circuit',
    *,
    patch: 'gen.Patch | gen.StabilizerCode | gen.ChunkInterface | dict[int, gen.Patch | gen.StabilizerCode | gen.ChunkInterface] | None' = None,
    tile_color_func: 'Callable[[gen.Tile], tuple[float, float, float, float] | tuple[float, float, float] | str] | None' = None,
    width: 'int' = 500,
    height: 'int' = 500,
    known_error: 'Iterable[stim.ExplainedError] | None' = None,
) -> 'str_html':
```

<a name="gen.stim_circuit_with_transformed_coords"></a>
```python
# gen.stim_circuit_with_transformed_coords

# (at top-level in the gen module)
def stim_circuit_with_transformed_coords(
    circuit: 'stim.Circuit',
    transform: 'Callable[[complex], complex]',
) -> 'stim.Circuit':
    """Returns an equivalent circuit, but with the qubit and detector position metadata modified.
    The "position" is assumed to be the first two coordinates. These are mapped to the real and
    imaginary values of a complex number which is then transformed.

    Note that `SHIFT_COORDS` instructions that modify the first two coordinates are not supported.
    This is because supporting them requires flattening loops, or promising that the given
    transformation is affine.

    Args:
        circuit: The circuit with qubits to reposition.
        transform: The transformation to apply to the positions. The positions are given one by one
            to this method, as complex numbers. The method returns the new complex number for the
            position.

    Returns:
        The transformed circuit.
    """
```

<a name="gen.stim_circuit_with_transformed_moments"></a>
```python
# gen.stim_circuit_with_transformed_moments

# (at top-level in the gen module)
def stim_circuit_with_transformed_moments(
    circuit: 'stim.Circuit',
    *,
    moment_func: 'Callable[[stim.Circuit], stim.Circuit]',
) -> 'stim.Circuit':
    """Applies a transformation to regions of a circuit separated by TICKs and blocks.

    For example, in this circuit:

        H 0
        X 0
        TICK

        H 1
        X 1
        REPEAT 100 {
            H 2
            X 2
        }
        H 3
        X 3

        TICK
        H 4
        X 4

    `moment_func` would be called five times, each time with one of the H and X instruction pairs.
    The result from the method would then be substituted into the circuit, replacing each of the H
    and X instruction pairs.

    Args:
        circuit: The circuit to return a transformed result of.
        moment_func: The transformation to apply to regions of the circuit. Returns a new circuit
            for the result.

    Returns:
        A transformed circuit.
    """
```

<a name="gen.str_html"></a>
```python
# gen.str_html

# (at top-level in the gen module)
class str_html:
    """A string that will display as an HTML widget in Jupyter notebooks.

    It's expected that the contents of the string will correspond to the
    contents of an HTML file.
    """
```

<a name="gen.str_html.write_to"></a>
```python
# gen.str_html.write_to

# (in class gen.str_html)
def write_to(
    self,
    path: 'str | pathlib.Path | io.IOBase',
):
    """Write the contents to a file, and announce that it was done.

    This method exists for quick debugging. In many contexts, such as
    in a bash terminal or in PyCharm, the printed path can be clicked
    on to open the file.
    """
```

<a name="gen.str_svg"></a>
```python
# gen.str_svg

# (at top-level in the gen module)
class str_svg:
    """A string that will display as an SVG image in Jupyter notebooks.

    It's expected that the contents of the string will correspond to the
    contents of an SVG file.
    """
```

<a name="gen.str_svg.write_to"></a>
```python
# gen.str_svg.write_to

# (in class gen.str_svg)
def write_to(
    self,
    path: 'str | pathlib.Path | io.IOBase',
):
    """Write the contents to a file, and announce that it was done.

    This method exists for quick debugging. In many contexts, such as
    in a bash terminal or in PyCharm, the printed path can be clicked
    on to open the file.
    """
```

<a name="gen.svg"></a>
```python
# gen.svg

# (at top-level in the gen module)
def svg(
    objects: 'Iterable[gen.Patch | gen.StabilizerCode | gen.ChunkInterface | stim.Circuit]',
    *,
    background: 'gen.Patch | gen.StabilizerCode | gen.ChunkInterface | stim.Circuit | None' = None,
    title: 'str | list[str] | None' = None,
    canvas_height: 'int | None' = None,
    show_order: 'bool' = False,
    show_obs: 'bool' = True,
    opacity: 'float' = 1,
    show_measure_qubits: 'bool' = True,
    show_data_qubits: 'bool' = False,
    system_qubits: 'Iterable[complex]' = (),
    show_all_qubits: 'bool' = False,
    extra_used_coords: 'Iterable[complex]' = (),
    show_coords: 'bool' = True,
    find_logical_err_max_weight: 'int | None' = None,
    rows: 'int | None' = None,
    cols: 'int | None' = None,
    tile_color_func: 'Callable[[gen.Tile], str | tuple[float, float, float] | tuple[float, float, float, float] | None] | None' = None,
    stabilizer_style: "Literal['polygon', 'circles'] | None" = 'polygon',
    observable_style: "Literal['label', 'polygon', 'circles']" = 'label',
    show_frames: 'bool' = True,
    pad: 'float | None' = None,
) -> 'gen.str_svg':
    """Returns an SVG image of the given objects.
    """
```

<a name="gen.transpile_to_z_basis_interaction_circuit"></a>
```python
# gen.transpile_to_z_basis_interaction_circuit

# (at top-level in the gen module)
def transpile_to_z_basis_interaction_circuit(
    circuit: 'stim.Circuit',
    *,
    is_entire_circuit: 'bool' = True,
) -> 'stim.Circuit':
    """Converts to a circuit using CZ, iSWAP, and MZZ as appropriate.

    This method mostly focuses on inserting single qubit rotations to convert
    interactions into their Z basis variant. It also does some optimizations
    that remove redundant rotations which would tend to be introduced by this
    process.
    """
```

<a name="gen.transversal_code_transition_chunks"></a>
```python
# gen.transversal_code_transition_chunks

# (at top-level in the gen module)
def transversal_code_transition_chunks(
    *,
    prev_code: 'StabilizerCode',
    next_code: 'StabilizerCode',
    measured: 'PauliMap',
    reset: 'PauliMap',
) -> 'tuple[Chunk, ChunkReflow, Chunk]':
```

<a name="gen.verify_distance_is_at_least_2"></a>
```python
# gen.verify_distance_is_at_least_2

# (at top-level in the gen module)
def verify_distance_is_at_least_2(
    obj: 'stim.Circuit | stim.DetectorErrorModel | StabilizerCode',
):
```

<a name="gen.verify_distance_is_at_least_3"></a>
```python
# gen.verify_distance_is_at_least_3

# (at top-level in the gen module)
def verify_distance_is_at_least_3(
    obj: 'stim.Circuit | stim.DetectorErrorModel | StabilizerCode',
):
```

<a name="gen.viz_3d_gltf_model_html"></a>
```python
# gen.viz_3d_gltf_model_html

# (at top-level in the gen module)
def viz_3d_gltf_model_html(
    model: 'pygltflib.GLTF2',
) -> 'str_html':
```

<a name="gen.xor_sorted"></a>
```python
# gen.xor_sorted

# (at top-level in the gen module)
def xor_sorted(
    vals: 'Iterable[TItem]',
    *,
    key: 'Callable[[TItem], Any]' = lambda e: e) -> list[TItem]:
    """Sorts items and then cancels pairs of equal items.

    An item will be in the result once if it appeared an odd number of times.
    An item won't be in the result if it appeared an even number of times.
    """
    result = sorted(vals, key=key)
    n = len(result)
    skipped = 0
    k = 0
    while k + 1 < n:
        if result[k] == result[k + 1]:
            skipped += 2
            k += 2
        else:
            result[k - skipped] = result[k]
            k += 1
    if k < n:
        result[k - skipped] = result[k]
    while skipped:
        result.pop()
        skipped -= 1
    return result,
) -> 'list[TItem]':
    """Sorts items and then cancels pairs of equal items.

    An item will be in the result once if it appeared an odd number of times.
    An item won't be in the result if it appeared an even number of times.
    """
```
