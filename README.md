# Zebrafish Modeling
A computational and data-driven framework for modeling and analyzing larval zebrafish behavior, combining experimental observations with physics-based simulations to uncover how sensory inputs and environmental boundaries shape collective motion.

This repository is built on research involving both wild-type zebrafish and TYR mutant (vision-impaired) zebrafish, aiming to dissect the role of visual and hydrodynamic cues in locomotion and group dynamics. By bridging behavioral biology with active matter physics, the tools and models here offer a systematic approach to understanding how fish interact with their environment and with each other.

## Core Features

- **Trajectory Analysis & Data Processing**
    - Preprocessing of high-resolution larval zebrafish tracking data.
    - Extraction of movement metrics such as velocity, acceleration, turning angles, and interaction distances.
    - Statistical comparison of behavioral metrics across developmental stages (7, 14, 21 dpf) and sensory conditions.
- **Behavioral Modeling & Simulation**
    - Implementation of agent-based models inspired by active matter and collective dynamics.
    - Parameter tuning and sensitivity analysis to replicate experimental patterns.
    - Simulation of fish responses to varying boundary geometries and obstacle configurations.
- **Visualization Tools**
    - Generation of speed distributions (e.g., violin and box plots) and trajectory heatmaps.
    - Real-time animation of simulated group behavior.
    - Comparative plots highlighting wild-type vs. blind fish dynamics.
- **Comparative Behavioral Studies**
    - Quantitative analysis of vision’s impact on exploratory behavior and boundary sensing.
    - Evaluation of group structure, alignment, and clustering patterns under sensory impairment.

## Research Context

Larval zebrafish integrate hydrodynamic and visual cues to navigate their environment, yet the developmental onset and mechanistic role of vision in shaping spatial exploration remain poorly understood. Here, we combine quantitative behavioral experimentation with computational modeling to understand how visual feedback influences boundary interactions in zebrafish larvae. We recorded individual trajectories of wild-type larvae in circular Petri dish arenas under different boundary conditions at various points in development. Using idtracker.ai for high-precision tracking, we transformed positions and velocities into radial and angular distributions. We employed kernel-smoothed two-sample Kolmogorov–Smirnov tests to detect shifts in spatial occupancy over development.

At 7 days post-fertilization (dpf), larvae exhibited uniform distributions across all arena types. By 14dpf, wild-type fish in clear arenas showed a significant migration toward the perimeter, an effect that intensified by 21dpf, where fish circled the boundary in sustained patterns. In half-sanded arenas, wild-type larvae developed strong symmetry-breaking, preferentially occupying the side where reflection was visible. In contrast, blind mutants maintained uniform distributions at all stages, indicating that visual input is necessary for boundary bias.

To examine underlying mechanisms, we extracted age-dependent speed distributions and implemented three simulation conditions: (1) a Brownian random walk model, (2) a probabilistic turning model featuring a tunable weight function over distance, and (3) a reflection-guided model using Snell’s law to sample turning directions toward perceived reflected light. The physics-based turning model accurately recapitulated edge preference in sanded arenas and the symmetry-breaking observed in half-sanded arenas, validating the hypothesis that vision-driven reflection sampling governs boundary navigation.

Our integrated pipeline combining controlled genetic and arena manipulations with rigorous statistical analysis and flexible computational models provides a framework for understanding sensorimotor integration in zebrafish and other small organisms. Through a simple, low-parameter model, we can recapitulate the various symmetry-breaking events that occur during the development of the zebrafish visual system. These findings explicate the developmental role of visual cues in spatial behavior and offer a modular platform for future studies linking neural circuitry to emergent movement patterns.

Download the full length senior thesis [here](Eric_Zhu_Senior_Thesis.pdf).
