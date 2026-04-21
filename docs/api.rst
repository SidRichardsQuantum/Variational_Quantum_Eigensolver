API Reference
=============

Primary Workflows
-----------------

These are the public entrypoints most users should start from.

.. autofunction:: vqe.run_vqe

.. autofunction:: qpe.run_qpe

.. autofunction:: qite.run_qite

.. autofunction:: qite.run_qrte


VQE Benchmarks And Comparisons
------------------------------

.. autofunction:: vqe.run_vqe_optimizer_comparison

.. autofunction:: vqe.run_vqe_ansatz_comparison

.. autofunction:: vqe.run_vqe_multi_seed_noise

.. autofunction:: vqe.run_vqe_geometry_scan

.. autofunction:: vqe.run_vqe_low_qubit_benchmark

.. autofunction:: vqe.run_vqe_mapping_comparison


Excited-State Methods
---------------------

.. autofunction:: vqe.run_adapt_vqe

.. autofunction:: vqe.run_ssvqe

.. autofunction:: vqe.run_vqd

.. autofunction:: vqe.run_qse

.. autofunction:: vqe.run_lr_vqe

.. autofunction:: vqe.run_eom_vqe

.. autofunction:: vqe.run_eom_qse


Problems And Hamiltonians
-------------------------

Use these helpers when you need to inspect or reuse the shared molecule,
geometry, active-space, and expert-mode resolution layer.

.. autoclass:: common.ResolvedProblem
   :members:

.. autofunction:: common.resolve_problem

.. autofunction:: common.build_hamiltonian

.. autofunction:: vqe.build_hamiltonian

.. autofunction:: qpe.build_hamiltonian

.. autofunction:: qite.build_hamiltonian

.. autofunction:: common.generate_geometry

.. autofunction:: common.get_molecule_config

.. autodata:: common.MOLECULES
   :annotation:


Ansatzes And Optimizers
-----------------------

.. autodata:: vqe.ANSATZES
   :annotation:

.. autofunction:: vqe.get_ansatz

.. autofunction:: vqe.init_params

.. autodata:: vqe.OPTIMIZERS
   :annotation:

.. autofunction:: vqe.get_optimizer

.. autofunction:: vqe.get_optimizer_stepsize


QPE Analysis Helpers
--------------------

.. autofunction:: qpe.bitstring_to_phase

.. autofunction:: qpe.phase_to_energy_unwrapped

.. autofunction:: qpe.hartree_fock_energy

.. autofunction:: common.qpe_branch_candidates

.. autofunction:: common.analyze_qpe_result

.. autofunction:: common.qpe_calibration_plan


Benchmark Utilities
-------------------

.. autofunction:: common.exact_ground_energy_for_problem

.. autofunction:: common.summarize_problem

.. autofunction:: common.summarize_registry_coverage

.. autofunction:: common.ionization_energy_panel

.. autofunction:: common.summary_stats

.. autofunction:: common.timed_call

.. autofunction:: common.compute_fidelity


Plotting
--------

.. autofunction:: vqe.plot_convergence

.. autofunction:: vqe.plot_optimizer_comparison

.. autofunction:: vqe.plot_ansatz_comparison

.. autofunction:: vqe.plot_noise_statistics

.. autofunction:: vqe.plot_multi_state_convergence

.. autofunction:: qpe.plot_qpe_distribution

.. autofunction:: qpe.plot_qpe_sweep

.. autofunction:: qite.plot_convergence

.. autofunction:: qite.plot_noise_statistics

.. autofunction:: qite.plot_diagnostics

.. autofunction:: common.plot_molecule

.. autofunction:: common.infer_bonds

.. autofunction:: common.infer_angles_from_bonds


Advanced I/O And Naming
-----------------------

These helpers are public for notebooks and reproducibility tooling, but most
users should prefer the high-level workflow functions above.

.. autofunction:: vqe.make_run_config_dict

.. autofunction:: vqe.run_signature

.. autofunction:: vqe.save_run_record

.. autofunction:: vqe.ensure_dirs

.. autofunction:: qpe.signature_hash

.. autofunction:: qpe.save_qpe_result

.. autofunction:: qpe.load_qpe_result

.. autofunction:: qpe.apply_noise_all

.. autofunction:: qite.make_run_config_dict

.. autofunction:: qite.run_signature

.. autofunction:: qite.save_run_record

.. autofunction:: qite.make_filename_prefix

.. autofunction:: qite.ensure_dirs

.. autofunction:: common.build_filename

.. autofunction:: common.save_plot

.. autofunction:: common.format_molecule_name

.. autofunction:: common.format_token


Advanced QITE Engine Utilities
------------------------------

These functions expose the lower-level VarQITE/VarQRTE engine used by the
workflow entrypoints.

.. autofunction:: qite.make_device

.. autofunction:: qite.make_energy_qnode

.. autofunction:: qite.make_state_qnode

.. autofunction:: qite.build_ansatz

.. autofunction:: qite.qite_step

.. autofunction:: qite.qrte_step
