====================================================================
ECHO QUBO OPTIMIZATION FRAMEWORK
====================================================================

ECHO: Eigenvalue-Guided Constrained Homotopy Optimization for QUBO
Experimental Framework for Regulated Health Insurance Plan Design


Author
--------------------------------------------------------------------
Firas F. Khabour
Innovestor DMCC
Dubai, UAE



====================================================================
1. PURPOSE OF THE REPOSITORY
====================================================================

This repository contains the complete experimental framework used in
the research study:

   "Combinatorial Optimization of Regulated Health Insurance Plan
    Design Using Quadratic Binary Programming"

The system implements a structured QUBO optimization model for
regulated health insurance product design and evaluates several
optimization methods.

Optimization Methods Implemented
--------------------------------

Baseline Methods
   • Greedy heuristic
   • Simulated Annealing (SA)
   • Exact optimization (Gurobi)

Proposed Method
   • ECHO — Eigenvalue-guided Constrained Homotopy Optimization

The repository enables full reproduction of the computational study
reported in the manuscript.

The experiments evaluate solver performance on:

   • 520 structured QUBO instances
   • Four insurance design scenarios
   • Multiple problem sizes

Project reference:
   ECHO for EJOR 2026


Research Objectives
-------------------

The study investigates two primary questions:

   • How spectral conditioning of the QUBO matrix affects
     heuristic optimization performance

   • Whether spectral continuation can improve solution quality
     under fixed computational budgets



====================================================================
2. CORE OPTIMIZATION CONCEPT
====================================================================

The insurance plan design problem is formulated as a Quadratic
Unconstrained Binary Optimization (QUBO) model.

Objective function:

        minimize   vᵀ Q v
        subject to v ∈ {0,1}ⁿ


Matrix decomposition:

        Q = Q_obj + Q_pen


Q_obj represents economic components:

   • expected healthcare cost
   • correlated claim risk
   • premium revenue


Q_pen represents feasibility penalties:

   • one-hot decision constraints
   • regulatory minimum coverage
   • affordability restrictions


As dimensionality increases, penalty scaling produces large eigenvalue
dispersion and high condition numbers. This creates ill-conditioned
optimization landscapes that are difficult for classical heuristics.

The ECHO algorithm addresses this using:

   • spectral diagnostics of Q
   • continuation over penalty intensity
   • condition-number guided beam search
   • roughness-weighted evaluation allocation



====================================================================
3. REPOSITORY STRUCTURE
====================================================================

Project Root

   insurance-quantum/

Main Files

   Start_ECHO_Experiment.py      Main experiment launcher
   README.txt                    Documentation
   Requirements.txt              Python dependency list
   project_dump.txt              Repository snapshot

Configuration

   config/
       config.yaml
       config_qubo_S1.yaml
       config_qubo_S2.yaml
       config_qubo_S3.yaml
       config_qubo_S4.yaml
       experiment_plan.yaml

Data

   data/
       seeds/

Source Code

   src_code/
       generators/
       solvers/
       runners/
       utils/

Results

   results/
       baseline_full_results.csv
       echo_full_results.csv

       maxcut/
       portfolio_card/
       spectral_dense/


The file "project_dump.txt" contains a snapshot of the repository
structure and source code for archival and reproducibility purposes.



====================================================================
4. SYSTEM REQUIREMENTS
====================================================================

Operating Systems Tested

   • Windows 10 / 11
   • Linux (Ubuntu recommended)


Python Version

   Python 3.10 or newer



====================================================================
5. REQUIRED SOFTWARE LIBRARIES
====================================================================

Required Python packages:

   numpy
   pandas
   pyyaml
   tqdm


Install dependencies:

   pip install -r Requirements.txt


Optional Solver

   Gurobi Optimizer

The program runs without Gurobi, but exact benchmark results will
not be available.



====================================================================
6. HARDWARE REQUIREMENTS
====================================================================

Recommended minimum system:

   CPU   : 8 cores
   RAM   : 16 GB
   Disk  : 5 GB free


Large experiments (N = 300) benefit from:

   CPU   : 16 cores
   RAM   : 32 GB



====================================================================
7. RUNNING THE PROGRAM
====================================================================

Launch the experiment controller:

   python Start_ECHO_Experiment.py

The program will open an interactive experiment menu.



====================================================================
8. MAIN MENU
====================================================================

   MAIN MENU

      1. Run Full Pipeline
      2. Run Primary QUBO Pipeline
      3. Run Benchmark QUBO Families
      Q. Quit



====================================================================
9. OPTION 1 — FULL PIPELINE
====================================================================

Runs the complete experiment workflow:

   • Synthetic data generation
   • QUBO construction
   • Greedy solver
   • Simulated annealing solver
   • Gurobi benchmark (if installed)
   • ECHO optimization
   • Benchmark QUBO families

This option produces all results used in the manuscript.



====================================================================
10. OPTION 2 — PRIMARY QUBO EXPERIMENT
====================================================================

Runs the insurance optimization study described in the paper.

Total instances solved:

   520


Scenario Definitions

   S1_cost_only        Expected cost minimization
   S2_risk_adjusted    Mean-variance risk objective
   S3_tight_regulation Strong regulatory penalties
   S4_affordability    Premium affordability restrictions


Instance Grid

   Problem sizes:

      N = 20, 30, 40, 50, 60, 100, 150, 200, 300

   Seeds:

      1000 – 1019


Instances per scenario:

   130

Total instances:

   520

Project reference:
   ECHO for EJOR 2026



====================================================================
11. OPTION 3 — BENCHMARK QUBO FAMILIES
====================================================================

Evaluates solver performance on standard QUBO families:

   • maxcut
   • portfolio_card
   • spectral_dense


Solvers used:

   • Simulated Annealing (SA)
   • ECHO-SA


Results stored in:

   results/maxcut
   results/portfolio_card
   results/spectral_dense



====================================================================
12. SOLVER DEFINITIONS
====================================================================

Greedy

   Multi-start greedy descent
   20 restarts

   Used as a baseline heuristic.


Simulated Annealing

   Multi-start simulated annealing

      steps per restart = 40,000
      restarts          = 20

   Total evaluations per instance:

      800,000


ECHO

   Eigenvalue-guided homotopy metaheuristic.

   Key mechanisms:

      • spectral landscape analysis
      • adaptive continuation stages
      • roughness-weighted evaluation allocation
      • condition-number guided beam search

   Parameters:

      maximum stages    = 8
      initial restarts  = 40
      beam size range   = 10 – 40



====================================================================
13. SYNTHETIC DATA GENERATION
====================================================================

Synthetic insurance datasets are generated using an actuarial
frequency-severity model.

Generation steps:

   1. Member risk scores from lognormal distribution
   2. Claim frequency from Poisson process
   3. Claim severity from lognormal distribution
   4. Claims assigned to 12 healthcare categories

This produces realistic cost structures and covariance matrices.

These values determine:

   • feature expected costs
   • risk covariance matrix



====================================================================
14. QUBO CONSTRUCTION
====================================================================

Decision variables:

   x_i   feature inclusion
   y_j   deductible band
   z_k   premium band
   t_l   regulatory slack variables


Constraints encoded as quadratic penalties:

   • one-hot premium band
   • one-hot deductible band
   • minimum regulatory coverage
   • affordability restrictions



====================================================================
15. OUTPUT FILES
====================================================================

baseline_full_results.csv

Results for:

   • Greedy
   • Simulated Annealing
   • Gurobi


Key columns:

   objective_raw_sa_best
   objective_raw_greedy
   objective_raw_gurobi

   energy_sa
   energy_greedy

   Q_condition_number
   Q_eigenvalue_min
   Q_eigenvalue_max



echo_full_results.csv

Results for ECHO.

Important fields:

   echo_raw_objective
   echo_energy
   echo_runtime_sec
   echo_num_stages
   echo_beam_size



====================================================================
16. RAW OBJECTIVE VS QUBO ENERGY
====================================================================

Two metrics are reported.

Raw objective

   objective_raw

Represents the economic objective:

   expected cost
   + risk
   − premium revenue

Used for solver comparison.


QUBO energy

   energy_total

Includes penalty contributions and is used internally by solvers.



====================================================================
17. FEASIBILITY METRICS
====================================================================

Solutions are validated against:

   • one-hot premium selection
   • one-hot deductible selection
   • regulatory constraints
   • affordability constraints

Final feasibility indicator:

   is_feasible



====================================================================
18. ENERGY DECOMPOSITION VALIDATION
====================================================================

For every solution vector:

        vᵀ Q v

is verified to equal:

        raw objective + penalty contributions

Solutions failing this validation are rejected.

No violations occurred across the full corpus.



====================================================================
19. BENCHMARK RESULT FILES
====================================================================

Each benchmark family produces:

   sa_results.csv
   sa_echo_results.csv
   results_master.csv

The file "results_master.csv" contains paired solver comparisons.

Metrics include:

   raw objective
   energy
   runtime
   condition number
   negative eigenvalue count



====================================================================
20. PREFLIGHT VALIDATION
====================================================================

Before each benchmark run the system verifies:

   • penalty scaling validity
   • annealing temperature calibration
   • acceptance ratio

Runs automatically abort if:

   • SA frozen
   • SA too hot
   • penalty scaling invalid

This protects experimental integrity.



====================================================================
21. REPRODUCIBILITY
====================================================================

Experiments are deterministic given:

   • random seed
   • scenario configuration
   • solver parameters

The system records:

   • QUBO matrices
   • index maps
   • run manifests
   • solver diagnostics



====================================================================
22. RECOMMENDED WORKFLOW
====================================================================

To reproduce the study:

   1. Install dependencies
   2. Run Start_ECHO_Experiment.py
   3. Select option 1 (Full Pipeline)
   4. Wait for experiments to complete
   5. Analyse the CSV outputs



====================================================================
CONTACT
====================================================================

Firas F. Khabour
Innovestor DMCC
Dubai, UAE