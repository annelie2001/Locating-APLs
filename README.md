# Locating Automated Parcel Lockers: A Simulation-Optimization Approach for the City of Würzburg

## Research Context

This repository contains the full implementation of the simulation-optimization framework developed for my seminar paper in the master's program Information Systems at the Chair of Business Analytics, University of Würzburg. The central research question is how to efficiently allocate APLs in a way that minimizes setup and delivery costs while satisfying uncertain, spatially distributed demand under capacity constraints. To this end, I follow an adapted version of the integrated simulation-optimization approach proposed by [Rabe et al. (2020)]([url](http://dx.doi.org/10.1109/WSC48552.2020.9384087)).

## Simulation-Optimization Framework

- 📦 Simulating multi-period demand using a System Dynamics (SD) model
- 📈 Generating stochastic demand scenarios based on the SD results
- 🗺️ Formulating and solving the Capacitated Facility Location Problem (CFLP) for the gen-
erated demand scenarios under different parameter settings
- 🔁 Evaluating the reliability of proposed APL placement strategies using Monte Carlo
Simulation (MCS)
- 🚚 Assessing variable costs of the proposed placement strategies by solving the corre-
sponding Capacitated Vehicle Routing Problem (CVRP)

## Repository Structure

├── 1_Data Generation/ # Marimo notebooks for creating or preprocessing datasets (e.g. calculating driving distances, generating demand scenarios,...)  
├── 2_Optimization/ # Marimo notebooks for solving the CFLP and generating the heuristic solution  
├── 3_Evaluation/ # Marimo notebooks for the MCS and the CVRP  
├── Data/ # All used and generated datasets (e.g. population, demand, coordinates,...)  
├── Images/ # Image files created for documentation  
├── Vensim Model/ # Vensim model and simulation files  
├── project-overview.py # Marimo notebook containing interactive maps and graphs across all project steps  

## Technologies Used

- Marimo – Used to build interactive notebooks for visualizing and analyzing results.
- IBM ILOG CPLEX – Applied to solve the CFLP.
- Google OR-Tools – Used for solving the CVRP based on real road distances.
- Docker – Employed to run a local OSRM (Open Source Routing Machine) server for calculating realistic walking and driving distances.


