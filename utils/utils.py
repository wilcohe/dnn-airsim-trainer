from arch.lipschitz import lcc_dense
from tensorflow import keras
import tensorflow as tf
import airsim
import numpy as np
import os, sys
import signal

def loadPath(path): 
	"""
	Load CSV Path
	input: path string
	"""

	if not os.path.exists(path): 
		print(f"Path {path} does not exist")
		sys.exit(-1)

	try: 
		paths = np.genfromtxt(path)
	except Exception as e: 
		raise e

	return paths


def reset():
	"""
	Reset airsim client
	"""

	client.reset()
	client.enableApiControl(True)
	client.armDisarm(True)


def parseState(state): 
	"""
	Parse KinematicsState Variable to DNN-friendly shape
	input: airsim KinematicsState type
	"""
	pos = state.kinematics_estimated.position.to_numpy_array().reshape(-1, 1)
	vel = state.kinematics_estimated.linear_velocity.to_numpy_array().reshape(-1, 1)
	attitude = np.array(airsim.utils.to_eularian_angles(state.kinematics_estimated.orientation)).reshape(-1, 1)
	rates = state.kinematics_estimated.angular_velocity.to_numpy_array().reshape(-1, 1)

	return np.vstack([state.timestamp, pos, vel, attitude, rates]).reshape(-1, 1)

def loadModel(path=""): 
	if not os.path.exists(path): 

		print(f"Path {path} does not exist")

		reg = lcc_dense(1, 16)["kernel_constraint"]

		# Generate model structure
		inp = keras.Input(shape=(24,))
		x = keras.layers.Dense(48, activation='tanh', kernel_constraint=reg)(inp)
		x = keras.layers.Dense(96, activation='relu', kernel_constraint=reg)(x)
		x = keras.layers.Dense(36, activation='tanh', kernel_constraint=reg)(x)
		x = keras.layers.Dense(12, activation='softmax', kernel_constraint=reg)(x)
		output_rpy = keras.layers.Dense(3, activation='tanh', kernel_constraint=reg)(x)
		output_t = keras.layers.Dense(1, activation='sigmoid', kernel_constraint=reg)(x)
		outputs = keras.layers.Concatenate()([output_rpy, output_t])

		# Define the loss function
		def cust_loss(y_waypoint, y_resultant): 
			position = y_waypoint[1:4] - y_resultant[0:3]
			heading = y_waypoint[7:10] - y_resultant[6:9]
			diff = tf.square(position) + tf.square(heading) + tf.square(y_waypoint[0])
			return tf.reduce_mean(diff, axis=-1)

		# Finish Model initialization
		model = keras.Model(inputs = inp, outputs=outputs)
		model.compile(optimizer='adam', loss = cust_loss)

		return model

	else:
		model = keras.models.load_model(path)
		return model

## Copied from: https://github.com/ryran/reboot-guard/blob/master/rguard#L284:L304

class GracefulDeath:
    """Catch signals to allow graceful shutdown."""

    def __init__(self):
        self.receivedSignal = self.receivedTermSignal = False
        catchSignals = [
            1,
            2,
            3,
            10,
            12,
            15,
        ]
        for signum in catchSignals:
            signal.signal(signum, self.handler)

    def handler(self, signum, frame):
        self.lastSignal = signum
        self.receivedSignal = True
        if signum in [2, 3, 15]:
            self.receivedTermSignal = True




