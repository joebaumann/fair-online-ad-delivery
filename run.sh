#!/bin/bash

# make bash script executable: chmod +x run.sh
# then run the simulation: ./run.sh

scenario="A"        # specify which experiment scenario to run (A, B, or C). The scenario must be present in the scenario_config.py file
path="results"      # specify the directory to store the results
seed_start=42       # the first seed to use. it will then be incremented for every run
nr_of_sim=30        # number of simulations to run for this scenario
nr_of_cpu_cores=1   # batch size, i.e., number of cpu cores to use in parallel 

# Run first batch of Python scripts with scenario passed as command line argument
echo "-- running scenario $scenario $nr_of_sim times with seeds starting from $seed_start --"
for ((seed=seed_start;seed<=seed_start+nr_of_sim-1;seed++));
do
    ((i=i%nr_of_cpu_cores)); ((i++==0)) && wait
    python run_simulation.py -p "$path" -s "$scenario" --seed $seed &
done

# Wait for all background jobs to finish before running second batch
wait
echo "-- DONE: scenario $scenario ran $nr_of_sim times --"

# Run second batch of Python scripts with scenario passed as command line argument
python plot_simulation_results.py -p "$path" -s "$scenario"
