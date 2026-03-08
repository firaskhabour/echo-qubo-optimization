====================================================================
ECHO QUBO OPTIMIZATION FRAMEWORK
====================================================================

Eigenvalue-Guided Homotopy Optimization 
for Quadratic Unconstrained Binary Optimization 
with Application to Health Insurance Plan Design


Author
--------------------------------------------------------------------
Eng. Firas F. Khabour
Innovestor DMCC
Dubai, UAE



====================================================================
1. PURPOSE OF THE REPOSITORY
====================================================================

This repository contains the complete experimental framework used in
the research study:

   "ECHO: Eigenvalue-guided Homotopy Optimization for QUBO
   with Application to Health Insurance Plan Design"

The system implements a structured QUBO optimization model for
regulated health insurance product design and evaluates several
optimization methods including a newly introduced ECHO method.

The system also benchmarks performance of ECHO and Simulated
Annealing (SA) on additional QUBO families.

The benchmark QUBO families implemented in the repository are:

   • MaxCut
   • Cardinality-constrained portfolio selection
   • Spectral-dense random QUBO generators


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

The experiments evaluate solver performance on two classes of
optimization problems:

   • Structured insurance QUBO instances (520 instances across
     four regulatory scenarios)

   • Independent QUBO benchmark families used to test the
     generality of the algorithm beyond the insurance domain.


These benchmark experiments demonstrate whether improvements obtained
by ECHO are driven by spectral structure of the QUBO matrix rather
than by domain-specific modelling.



Research Objectives
-------------------

The study investigates four primary questions:

• Conditioning-aware QUBO metaheuristics.
• Empirical linkage between spectral conditioning and heuristic performance.
• A structured QUBO formulation for regulated health insurance plan design.
• Evidence on when spectral continuation improves optimisation.




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
   • homotopy continuation over penalty intensity
   • condition-number guided search
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
       insurance_baseline_results.csv
       insurance_echo_results_master.csv

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
      
      1. Run Full Pipeline (option 2 & option 3)"
      2. Run Primary QUBO Pipeline (Generate instances → Greedy → SA → Gurobi → ECHO-SA)"
      3. Run Benchmark QUBO Families (Max-Cut, Portfolio, Spectral_dense) → SA → ECHO-SA"
      Q. Quit"


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
10. OPTION 2 — INSURANCE QUBO EXPERIMENT
====================================================================

Runs the structured insurance QUBO study described in the paper.

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



====================================================================
11. OPTION 3 — BENCHMARK QUBO FAMILIES
====================================================================

This option evaluates solver performance on independent QUBO families
used in the benchmark study of the paper.

These experiments test whether the spectral continuation mechanism of
ECHO generalizes beyond the insurance design application.

Three benchmark QUBO families are implemented.

maxcut
   Graph partitioning QUBO instances generated from random graphs.

portfolio_card
   Mean–variance portfolio optimization with cardinality constraints.

spectral_dense
   Random dense QUBO matrices with controlled eigenvalue dispersion.


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

      • spectral diagnostics of the QUBO matrix
      • homotopy continuation over penalty curvature
      • staged search across continuation levels
      • adaptive allocation of evaluation budget

   Parameters:

      maximum stages    = 8
      initial restarts  = 40



====================================================================
13. SYNTHETIC DATA GENERATION
====================================================================

Synthetic insurance datasets are generated using an actuarial
frequency-severity model.

Generation steps:

   1. Member risk scores from lognormal distribution
   2. Claim frequency from Poisson process
   3. Claim severity from lognormal distribution
   4. Claims assigned to healthcare service categories

This produces realistic cost structures and covariance matrices.



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

insurance_baseline_results.csv

Results for:

   • Greedy
   • Simulated Annealing
   • Gurobi


insurance_echo_results_master.csv

Results for ECHO optimization.


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


QUBO energy

   energy_total

Includes penalty contributions and is used internally by solvers.



====================================================================
17. BENCHMARK RESULT FILES
====================================================================

Each benchmark family produces:

   maxcut_sa_results.csv
   maxcut_echo_results.csv
   maxcut_results_master.csv

   portfolio_card_sa_results.csv
   portfolio_card_echo_results.csv
   portfolio_card_results_master.csv

   spectral_dense_sa_results.csv
   spectral_dense_echo_results.csv
   spectral_dense_results_master.csv



====================================================================
18. PREFLIGHT VALIDATION
====================================================================

Before each benchmark run the system verifies:

   • penalty scaling validity
   • annealing temperature calibration
   • acceptance ratio



====================================================================
19. REPRODUCIBILITY
====================================================================

Experiments are deterministic given:

   • random seed
   • scenario configuration
   • solver parameters



====================================================================
20. RECOMMENDED WORKFLOW
====================================================================

To reproduce the study:

   1. Install dependencies
   2. Run Start_ECHO_Experiment.py
   3. Select option 1 (Full Pipeline)
   4. Wait for experiments to complete
   5. Analyse the CSV outputs

====================================================================
21. ROBUSTNESS TEST — AUTO TEMPERATURE MODE
====================================================================

The baseline insurance experiments use a fixed simulated annealing
temperature schedule:

      T0   = 5
      Tend = 0.01

To verify that the SA vs ECHO comparison is not dependent on this
fixed schedule, a robustness experiment can be executed using
automatic temperature calibration.

Recommended robustness setup:

      N = 300
      Scenarios = S1, S2, S3, S4
      Seeds = 2000 – 2009

PowerShell run command:

   $env:SA_AUTO_TEMPERATURE="True"
   python Start_ECHO_Experiment.py

   For limited run to Robustness as per paper: 
   python src_code/runners/run_baseline_full.py
    --robustness; python src_code/runners/run_echo_full.py 
    --baseline results/insurance_robustness_baseline_results.csv 
    --output results/robustness_results_master_InsuranceQubo.csv 
    --robustness_seeds

Then run the insurance robustness subset for:

   • N = 300
   • S1, S2, S3, S4
   • seeds 2000–2009

If running directly through runner scripts, use:

   $env:SA_AUTO_TEMPERATURE="True"
   python src_code\runners\run_baseline_full.py
   python src_code\runners\run_echo_full.py

Solvers executed:

   • Simulated Annealing (SA)
   • ECHO-SA

Expected output:

   robustness_results_master.csv

This experiment confirms that the relative performance conclusions
reported in the manuscript remain consistent when the annealing
temperature is scaled automatically with the QUBO energy magnitude.

====================================================================
22. TABLES DIRECTORY
====================================================================

The repository includes a folder containing machine-readable tables
that document the experimental configuration used in the study.

Folder location:

   tables/

These files provide structured descriptions of solver settings,
QUBO model parameters, and experiment grids used to generate the
results reported in the manuscript.

--------------------------------------------------------------------
Machine-readable tables
--------------------------------------------------------------------

solver_parameters_machine.csv

   Solver configuration parameters including evaluation budgets,
   restart policies, and temperature schedules for:

      • Greedy
      • Simulated Annealing (SA)
      • ECHO-SA

insurance_qubo_parameters_machine.csv

   Parameters used in constructing the structured insurance QUBO
   model, including penalty scaling and decision variable structure.

insurance_experiment_grid_machine.csv

   Defines the experiment grid for the insurance QUBO study
   including scenario definitions, problem sizes, and seed ranges.

insurance_experiment_grid_detailed_machine.csv

   Expanded instance list where each row represents one experiment
   defined by (scenario, N, seed).

benchmark_default_grid_machine.csv

   Default experiment grids used for benchmark QUBO families
   (MaxCut, Portfolio, Spectral-dense).

benchmark_generator_parameters_machine.csv

   Parameters used when generating benchmark QUBO instances.

--------------------------------------------------------------------
Human-readable reference
--------------------------------------------------------------------

parameter_reference_human.txt

   A formatted summary of the experimental configuration provided
   for quick reference.

--------------------------------------------------------------------
Purpose
--------------------------------------------------------------------

These tables provide transparency and reproducibility of the
experimental setup without requiring inspection of the source code.

====================================================================
23. REPLICATION PACKAGE
====================================================================

This repository serves as the replication package for the research
article:

   "ECHO: Eigenvalue-guided Homotopy Optimization for QUBO
   with Application to Health Insurance Plan Design"

The repository contains:

   • full experiment source code
   • configuration files
   • experiment parameter tables
   • benchmark generators
   • solver implementations
   • experiment runners
   • output result files

Running the experiment pipeline reproduces the computational results
reported in the manuscript.

All experiments are deterministic given:

   • random seed
   • scenario configuration
   • solver parameters

The repository therefore enables full reproduction of the study.

====================================================================
24. ARCHIVED VERSION (DOI)
====================================================================

Permanent archive of this repository:

https://doi.org/10.6084/m9.figshare.31562455

====================================================================
25. CONTACT
====================================================================

Firas F. Khabour
Innovestor DMCC
Dubai, UAE
firas.khabour@gmail.com