#!/bin/bash
# This script runs evolution with multiple population with same parameters.
# Example: ./multiple.sh /tmp/1Dpopulation /tmp/2Dpopulation

fitness="fitness/test.py"
algorithm="hillclimbing"
args="epoch_count=50 spawn_range=0.5"

for population in "$@"; do
	echo $population
	./evolution --run $fitness $population $algorithm $args > ${population}_solved
	./evolution --animate $fitness ${population}_solved
done
