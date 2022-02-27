#!/bin/bash

exit_gracefully(){
	kill -n 9 $PYTHON_PID
	kill -n 9 $AIRSIM_PID
	exit 0
}

# https://cravencode.com/post/essentials/parallel-commands-bash-scripts/

cd ~/Documents/Simulators/Airsim/<etc>

<path to launch airsim > & 

AIRSIM_PID=$!

wait $AIRSIM_PID

python3 ../IRALScripts/control_model/unsupervised_training.py ../IRALScripts/control/model/models/unsupervised_model ../IRALScripts/paths/<path> & 

PYTHON_PID=$!

wait $PYTHON_PID

# Command to kick off airsim (async)?
# wait
# Handshake with airsim 
# Kick off training script (async)

while :
do

	if ps -p $AIRSIM_PID > /dev/null
	then
		sleep 1
	else
		kill -n 9 $PYTHON_PID

		<path to launch airsim > & 

		AIRSIM_PID=$!

		wait $AIRSIM_PID

		python3 ../IRALScripts/unsupervised_training.py ../IRALScripts/models/unsupervised_model ../IRALScripts/paths/<path> & 

		PYTHON_PID=$!

		wait $PYTHON_PID

		trap exit_gracefully SIGINT
	fi

	# Check airsim heartbeat
	# If heartbeat false
		# Save model weights 
		# Safely kill training script
		# Kick off airsim
		# Handshake airsim

done