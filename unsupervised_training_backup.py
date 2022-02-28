from arch.lipschitz import lcc_dense
from tensorflow import keras
import tensorflow as tf
import airsim
import numpy as np
from utils.utils import *
import sys

def main(): 

	## Initialize Model
    model = loadModel(sys.argv[1])

	## Path loading
    path = loadPath(sys.argv[2])

	## Initialize Simulation
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.simPause(True)

    killer = GracefulDeath()


    while True: 

		## Initial State 
    	state = client.getMultirotorState()
	state_vec = parseState(state)	

        for line in path: 

		dnn_input = np.vstack([state_vec, np.array(line).reshape(-1, 1)]).reshape(-1, 1); print(line)

		ctrl = model(dnn_input, training=True)
		client.simPause(False)
		client.moveByAngleRatesThrottleAsync(ctrl[0], ctrl[1], ctrl[2], ctrl[3], 0.01)
		state = client.getMultirotorState()
		client.simPause(True)
		state_vec = parseState(state)


		if state.collision.has_collided: 
				## Measure velocity and do collision based on a fixed collision 
		    reset()
		    break

                loss = cust_loss(state_vec, line.reshape(-1, 1))

		with tf.GradientTape() as tape: 
		    grad = tape.gradient(loss, model.trainable_weights)

		optimizer.apply_gradients(zip(grads, model.trainable_weights))

		if killer.received_signal: 
			model.save(path)
			sys.exit(0)

if __name__ == "__main__": 
	main()



