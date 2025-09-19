#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 <threads> <indx_offset> <simulations>"
    echo "All three arguments must be integers."
    echo "threads: number of threads to use for simulations"
    echo "indx_offset: offset for the pid of the first thread"
    echo "simulations: number of simulations to run"
}

# Check if exactly 3 arguments are provided
if [ "$#" -ne 3 ]; then
    show_help
    exit 1
fi

# Check if all arguments are integers
for arg in "$@"; do
    if ! [[ "$arg" =~ ^-?[0-9]+$ ]]; then
        show_help
        exit 1
    fi
done

# Print the integers
echo "Starting $1 threads with offset $2 for $3 simulations"
simulations_per_thread=$(( $3 / $1 ))
echo "Each thread will run $simulations_per_thread simulations"
for ((i=0; i<$1; i++)); do
    idx=$((i * simulations_per_thread))
    sims_to_do=$simulations_per_thread
    if [ $i -eq $(( $1 - 1 )) ]; then
        sims_to_do=0
    fi

    echo "Starting thread $i with idx $idx and count $sims_to_do"
    # python src/simulator/simulate.py $sims_to_do $idx $((i + $2)) > "/data/lkolmar/logs/simulation_thread_${i}.log" 2>&1 &
    python src/simulator/simulate.py $sims_to_do $idx $((i + $2)) > "/data/lkolmar/logs/simulation_thread_${i}.log" 2>&1 &
done
wait