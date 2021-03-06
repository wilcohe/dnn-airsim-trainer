#!/bin/bash

exit_gracefully(){
	kill -n 9 $PYTHON_PID
	kill -n 9 $AIRSIM_PID
	exit 0
}

# https://cravencode.com/post/essentials/parallel-commands-bash-scripts/

cd ~/Documents/Simulators/Airsim/docker

echo $(pwd)

./run_airsim_image_binary.sh airsim_binary:11.0-devel-ubuntu20.04 Blocks/Blocks.sh -windowed -resX=1080 -resY=720 --settings ./settings.json & 

sleep 10

AIRSIM_PID=$!

wait $AIRSIM_PID

python3 ../IRALScripts/control_model/unsupervised_training.py ../IRALScripts/control_model/models/unsupervised_model ../IRALScripts/control_model/paths/squiggle_1.csv & 

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

		./run_airsim_image_binary.sh airsim_binary:11.0-devel-ubuntu20.04 Blocks/Blocks.sh -windowed -resX=1080 -resY=720 --settings ./settings.json & 

		AIRSIM_PID=$!

		wait $AIRSIM_PID

		python3 ../IRALScripts/control_model/unsupervised_training.py ../IRALScripts/control_model/models/unsupervised_model ../IRALScripts/control_model/paths/squiggle_1.csv & 

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
