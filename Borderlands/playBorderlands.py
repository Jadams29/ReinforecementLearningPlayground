import os
import os
import pickle
import sys
import time
import pandas as pd
import cv2
import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.utils
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

sys.path.append("C:\\Users\\joshu\\OneDrive - Georgia Institute of Technology\\ReinforecementLearningPlayground\\")

from HelperFunctions import Network
from borderlandsScienceEnvironment import BorderlandsScienceEnvironment


class Agent:
	def __init__(self, windowName, pathToExe, testing, epsilon=0.99,
	             epsilonDecay=0.001, alpha=1e-3, alphaDecay=1e-5,
	             nnLearningRate=1e-6, gamma=0.99, gammaDecay=1e-5,
	             useFrameSkip=True, numEpisodes=1001, verbose=True, startingWeights=None,
	             epsilonDecaySchedule="exponential", useTargetNetwork=True, decayEpsilon=False, saveVideo=False,
	             optimizer=None, saveDir=None, batchSize=32, useLeNet=False):
		self.UseLeNet = useLeNet
		self.OriginalDimensions = None
		self.Testing = testing
		# LoadX360CE.LoadX360CE(pathToExe)
		self.Actions = None
		self.ActionValueModel = None
		self.Alpha = None
		self.AlphaDecay = None
		self.Avg = None
		self.BatchSize = None
		self.CurrentEpisode = None
		self.CurrentState = None
		self.DataIDX = None
		self.DecayEpsilon = None
		self.EncodedRewardLength = None
		self.Environment = None
		self.Epsilon = None
		self.EpsilonDecay = None
		self.EpsilonDecaySchedule = None
		self.ExperienceReplay = None
		self.ExperienceReplayMaxSize = None
		self.ExperienceReplayRowSize = None
		self.FramesSinceLastSkip = None
		self.Gamma = None
		self.GammaDecay = None
		self.LastAction = None
		self.LearningRateForNetwork = None
		self.MinEpsilon = None
		self.ModelType = None
		self.Network = None
		self.NNInputShape = None
		self.NNOutputShape = None
		self.NumActions = None
		self.NumEpisodes = None
		self.NumEpochs = None
		self.NumSkipFrames = None
		self.Optimizer = None
		self.OriginalAlpha = None
		self.OriginalEpsilon = None
		self.OriginalGamma = None
		self.PreviousReplayMemoryIdx = None
		self.ResetEpsilon = None
		self.RoundLimit = None
		self.RunTimeReplay = None
		self.SampleSize = None
		self.SaveDirectory = None
		self.SaveEveryNFrame = None
		self.SaveVideo = None
		self.StartingWeights = None
		self.StateSize = None
		self.StateSize = None
		self.StepLimit = None
		self.StepsToMergeQ = None
		self.TargetActionValueModel = None
		self.TestEveryNEpisodes = None
		self.TestSize = None
		self.TotalRewardForEpisode = None
		self.UseFrameSkip = None
		self.UseRound = None
		self.UseTargetNetwork = None
		self.Verbose = None
		self.Video = None
		self.RewardsDuringTraining = None
		self.BestWeights = None
		self.BestReward = 0
		self.Setup(windowName=windowName, pathToExe=pathToExe, testing=testing, epsilon=epsilon,
		           epsilonDecay=epsilonDecay, alpha=alpha, alphaDecay=alphaDecay,
		           nnLearningRate=nnLearningRate, gamma=gamma, gammaDecay=gammaDecay,
		           useFrameSkip=useFrameSkip, numEpisodes=numEpisodes, verbose=verbose, startingWeights=startingWeights,
		           epsilonDecaySchedule=epsilonDecaySchedule, useTargetNetwork=useTargetNetwork,
		           decayEpsilon=decayEpsilon, saveVideo=saveVideo,
		           optimizer=optimizer, saveDir=saveDir, batchSize=batchSize)
	
	def Setup(self, windowName, pathToExe, testing, epsilon=0.99,
	          epsilonDecay=0.001, alpha=1e-3, alphaDecay=1e-5,
	          nnLearningRate=1e-6, gamma=0.99, gammaDecay=1e-5,
	          useFrameSkip=True, numEpisodes=1001, verbose=True, startingWeights=None,
	          epsilonDecaySchedule="exponential", useTargetNetwork=True, decayEpsilon=False, saveVideo=False,
	          optimizer=None, saveDir=None, batchSize=32):
		
		# This will reset all details
		self.Testing = testing
		# LoadX360CE.LoadX360CE(pathToExe)
		if not self.Testing:
			print("Load Xbox Streaming App and Connect to Xbox")
			self.Environment = BorderlandsScienceEnvironment(windowName=windowName, testing=False)
		else:
			self.Environment = gym.make("MsPacman-v0")
		
		initialFrame = self.Environment.reset()
		if not self.UseLeNet:
			initialFrame = cv2.cvtColor(initialFrame, cv2.COLOR_BGR2GRAY)
		self.OriginalDimensions = initialFrame.shape
		self.StartingWeights = startingWeights
		self.Optimizer = optimizer
		self.SaveDirectory = saveDir
		self.UseTargetNetwork = useTargetNetwork
		self.ResetEpsilon = False
		self.TestEveryNEpisodes = 10
		self.Actions = [i for i in range(self.Environment.action_space.n)]
		self.SaveEveryNFrame = 4
		self.NumActions = self.Environment.action_space.n
		self.NumEpisodes = numEpisodes
		self.StateSize = initialFrame.size
		self.DecayEpsilon = decayEpsilon
		self.EpsilonDecaySchedule = epsilonDecaySchedule
		self.Epsilon = epsilon
		self.EpsilonDecay = epsilonDecay
		self.StepLimit = 1000 if self.Testing else 100
		self.NNInputShape = initialFrame.size
		self.NNOutputShape = self.Environment.action_space.n
		self.MinEpsilon = 0.001 if self.Testing else 0.01
		self.OriginalEpsilon = np.copy(self.Epsilon)
		self.Alpha = alpha
		self.Avg = 0
		self.OriginalAlpha = np.copy(self.Alpha)
		self.AlphaDecay = alphaDecay
		self.Gamma = gamma
		self.OriginalGamma = np.copy(self.Gamma)
		self.GammaDecay = gammaDecay
		self.ModelType = None
		self.UseRound = False
		self.RoundLimit = 5
		self.CurrentEpisode = None
		self.EncodedRewardLength = 4
		self.LearningRateForNetwork = nnLearningRate
		self.StateSize = initialFrame.size
		# Establish the size of the Experience Replay
		#   Each row will have the currentState(w*h), action(1), reward(1), nextState(w*h), isDone(1)
		self.ExperienceReplayRowSize = (self.StateSize * 2) + 3
		self.ExperienceReplayMaxSize = 10000 if self.Testing else 1000
		self.ExperienceReplay = None
		self.PreviousReplayMemoryIdx = None
		self.CurrentState = None
		self.StepsToMergeQ = 1
		self.NumEpochs = 50
		self.BatchSize = batchSize
		self.TestSize = 10
		self.SampleSize = batchSize
		self.UseFrameSkip = useFrameSkip
		self.NumSkipFrames = 2
		self.FramesSinceLastSkip = 0
		self.LastAction = None
		self.Verbose = verbose
		self.Network = Network.ConvolutionalNeuralNetwork(output_size=self.NNOutputShape)
		self.RewardsDuringTraining = np.zeros(shape=((self.NumEpisodes // self.TestEveryNEpisodes) + 1, self.TestSize))
		self.DataIDX = {"State": {"Start": 0, "Stop": self.StateSize},
		                "Action": self.StateSize,
		                "Reward": self.StateSize + 1,
		                "NextState": {"Start": self.StateSize + 2, "Stop": self.StateSize * 2 + 2},
		                "IsNextStateTerminated": self.StateSize * 2 + 2}
		
		if self.StartingWeights is not None:
			self.loadWeights()
		else:
			self.ActionValueModel = self.createNeuralNetwork()
		self.TargetActionValueModel = self.createNeuralNetwork()
		plot_model(self.ActionValueModel, to_file="model2.png")
		
		# Initialize Replay
		self.ExperienceReplay = np.zeros(shape=(self.ExperienceReplayMaxSize, self.ExperienceReplayRowSize))
		
		# Initialize Replay With Random Values
		self.initializeExperienceReplay(numObservations=1000)
		self.RunTimeReplay = np.zeros(shape=(self.NumEpisodes, 1))
		self.TotalRewardForEpisode = np.zeros(shape=(self.NumEpisodes, 1))
		self.SaveVideo = saveVideo
	
	def loadWeights(self):
		self.ActionValueModel = tf.keras.models.load_model(self.StartingWeights)
	
	def epsilonDecayHelper(self):
		if self.EpsilonDecaySchedule.lower() == "linear":
			self.Epsilon -= self.EpsilonDecay
			self.validateEpsilon()
		elif self.EpsilonDecaySchedule.lower() == "exponential":
			self.Epsilon *= (1 - self.EpsilonDecay)
			self.validateEpsilon()
		elif self.EpsilonDecaySchedule.lower() == "custom":
			self.validateEpsilon()
	
	def validateEpsilon(self):
		if self.Epsilon < self.MinEpsilon:
			if self.ResetEpsilon:
				self.Epsilon = self.OriginalEpsilon
			else:
				self.Epsilon = 0.01
	
	def createNeuralNetwork(self):
		if self.UseLeNet:
			model = self.Network.create_model(input_shape=self.OriginalDimensions)
			if self.Optimizer.lower() == "adam":
				model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=self.LearningRateForNetwork),
				              loss=tensorflow.losses.mean_squared_error)
			elif self.Optimizer.lower() == "rmsprop":
				model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.LearningRateForNetwork),
				              loss=tensorflow.losses.mean_squared_error)
			elif self.Optimizer.lower() == "ada":
				model.compile(optimizer=tensorflow.keras.optimizers.Adagrad(learning_rate=self.LearningRateForNetwork),
				              loss=tensorflow.losses.mean_squared_error)
			return model
		else:
			X_in = Input(self.NNInputShape)
			
			# X = Activation('relu')(X_in)
			X = Dense(256, activation='relu', name='fully_connected_one')(X_in)
			# X = Dropout(0.1)(X)
			
			# X = Activation('relu')(X)
			X = Dense(512, activation='relu', name='fully_connected_two')(X)
			
			# X = Activation('relu')(X)
			X = Dense(1024, activation='relu', name='fully_connected_three')(X)
			# X = Dropout(0.2)(X)
			
			X = Dense(256, activation='relu', name='fully_connected_four')(X)
			
			X = Dense(64, activation='relu', name='fully_connected_five')(X)
			
			X = Dense(self.NNOutputShape, activation='linear', name='final_fully_connected')(X)
			
			model = Model(inputs=X_in, outputs=X, name='NeuralNetwork')
			# tensorflow.keras.optimizers.RMSprop
			if self.Optimizer.lower() == "adam":
				model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=self.LearningRateForNetwork),
				              loss=tensorflow.losses.mean_squared_error)
			elif self.Optimizer.lower() == "rmsprop":
				model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=self.LearningRateForNetwork),
				              loss=tensorflow.losses.mean_squared_error)
			elif self.Optimizer.lower() == "ada":
				model.compile(optimizer=tensorflow.keras.optimizers.Adagrad(learning_rate=self.LearningRateForNetwork),
				              loss=tensorflow.losses.mean_squared_error)
			return model
	
	def cleanUp(self):
		self.Video.release()
		cv2.destroyAllWindows()
		self.Video = None
	
	def initializeExperienceReplay(self, numObservations=100):
		print("\t\tInitializing Experience Replay")
		if self.StartingWeights is not None:
			with open("./experienceReplay.pkl", "rb") as inputFile:
				self.ExperienceReplay = pickle.load(inputFile)
		else:
			finishedInitializingExperienceReplay = False
			totalSteps = 0
			while not finishedInitializingExperienceReplay:
				terminated = False
				tState = self.Environment.reset()
				if self.UseLeNet:
					state = tState[:, :, ::-1]
				else:
					state = cv2.cvtColor(tState[:, :, ::-1], cv2.COLOR_BGR2GRAY)
				self.CurrentState = state
				episodeReward = 0
				while not terminated:
					action = self.getAction(state=state, onlyExploit=True)
					nextState, reward, terminated, _ = self.Environment.step(action)
					if self.UseLeNet:
						nextState = nextState[:, :, ::-1]
					else:
						nextState = cv2.cvtColor(nextState[:, :, ::-1], cv2.COLOR_BGR2GRAY)
					episodeReward += reward
					self.addToMemoryReplay(self.createMemory(state=state, action=action, reward=reward,
					                                         nextState=nextState, isNextStateTerminal=terminated))
					totalSteps += 1
					self.CurrentState = nextState
					state = nextState
					if totalSteps >= numObservations:
						terminated = True
						finishedInitializingExperienceReplay = True
		print("\t\tFinished Initializing Experience Replay")
	
	def sampleFromExperience(self, sampleSize=32):
		if self.PreviousReplayMemoryIdx > sampleSize:
			randIdx = np.random.choice(self.PreviousReplayMemoryIdx, size=sampleSize, replace=False)
		else:
			randIdx = np.random.choice(self.PreviousReplayMemoryIdx, size=sampleSize, replace=True)
		return self.ExperienceReplay[randIdx, :]
	
	def determineTarget(self, batch):
		currentStates = batch[:, self.DataIDX["State"]["Start"]:self.DataIDX["State"]["Stop"]]
		currentActions = batch[:, self.DataIDX["Action"]].astype(int)
		currentRewards = batch[:, self.DataIDX["Reward"]].astype(int)
		nextStates = batch[:, self.DataIDX["NextState"]["Start"]:self.DataIDX["NextState"]["Stop"]]
		terminalIdx = batch[:, self.DataIDX["IsNextStateTerminated"]] == 1
		nonTerminalIdx = batch[:, self.DataIDX["IsNextStateTerminated"]] != 1
		if self.UseLeNet:
			predY = self.ActionValueModel.predict(np.reshape(currentStates, newshape=(self.BatchSize,
			                                                                          self.OriginalDimensions[0],
			                                                                          self.OriginalDimensions[1],
			                                                                          self.OriginalDimensions[2])))
			targetPredY = np.max(self.TargetActionValueModel.predict(np.reshape(nextStates, newshape=(self.BatchSize,
			                                                                          self.OriginalDimensions[0],
			                                                                          self.OriginalDimensions[1],
			                                                                          self.OriginalDimensions[2])
			                                                                    )), axis=1)
			predY[nonTerminalIdx, currentActions[nonTerminalIdx]] = \
				currentRewards[nonTerminalIdx] + (self.Gamma * targetPredY[nonTerminalIdx])
			predY[terminalIdx, currentActions[terminalIdx]] = currentRewards[terminalIdx]
		else:
			predY = self.ActionValueModel.predict(currentStates)
			targetPredY = np.max(self.TargetActionValueModel.predict(nextStates), axis=1)
			predY[nonTerminalIdx, currentActions[nonTerminalIdx]] = \
				((1 - self.Gamma) * currentRewards[nonTerminalIdx]) + (self.Gamma * targetPredY[nonTerminalIdx])
			predY[terminalIdx, currentActions[terminalIdx]] = currentRewards[terminalIdx]

		return predY
	
	def getAction(self, state=None, onlyExploit=False):
		try:
			if self.FramesSinceLastSkip != 0:
				self.FramesSinceLastSkip += 1
				if self.FramesSinceLastSkip >= self.NumSkipFrames:
					self.FramesSinceLastSkip = 0
				return self.LastAction
			else:
				self.FramesSinceLastSkip += 1
				
				"""
				Use Epsilon Greedy
				"""
				if self.UseLeNet:
					if onlyExploit:
						self.LastAction = np.argmax(
							self.ActionValueModel.predict(np.expand_dims(state, axis=0)))
						return self.LastAction
					else:
						if np.random.rand() <= self.Epsilon:
							self.LastAction = np.random.choice(self.NumActions)
							return self.LastAction
						else:
							self.LastAction = np.argmax(
								self.ActionValueModel.predict(np.expand_dims(state, axis=0)))
							return self.LastAction
				else:
					if onlyExploit:
						self.LastAction = np.argmax(
							self.ActionValueModel.predict(np.reshape(state, newshape=(1, state.size))))
						return self.LastAction
					else:
						if np.random.rand() <= self.Epsilon:
							self.LastAction = np.random.choice(self.NumActions)
							return self.LastAction
						else:
							self.LastAction = np.argmax(
								self.ActionValueModel.predict(np.reshape(state, newshape=(1, state.size))))
							return self.LastAction
		except Exception as _exception:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(f"Exception in ''", _exception)
	
	def splitXAndY(self, data):
		X = data[:, :-self.NumActions]
		y = data[:, -self.NumActions:]
		return X, y
	
	def createMemory(self, state, action, reward, nextState, isNextStateTerminal):
		state = state.flatten()
		nextState = nextState.flatten()
		memory = np.zeros(shape=(self.ExperienceReplayRowSize,))
		memory[self.DataIDX["State"]["Start"]:self.DataIDX["State"]["Stop"]] = state
		memory[self.DataIDX["Action"]] = action
		memory[self.DataIDX["Reward"]] = reward
		memory[self.DataIDX["NextState"]["Start"]:self.DataIDX["NextState"]["Stop"]] = nextState
		memory[-1] = isNextStateTerminal
		return memory
	
	def addToMemoryReplay(self, memory):
		if self.PreviousReplayMemoryIdx is None:
			self.PreviousReplayMemoryIdx = 0
		elif self.PreviousReplayMemoryIdx >= self.ExperienceReplayMaxSize:
			self.PreviousReplayMemoryIdx = 0
		self.ExperienceReplay[self.PreviousReplayMemoryIdx, :] = memory
		self.PreviousReplayMemoryIdx += 1
	
	def getTrainingData(self, episodes=1, maxSteps=1000, saveVideo=False):
		try:
			print("\t\tGetting Data")
			all_episodes = []
			tempRewardResults = np.zeros(shape=(self.TestSize,))
			for episode in range(episodes):
				terminated = False
				tState = self.Environment.reset()
				if self.UseLeNet:
					state = tState[:, :, ::-1]
				else:
					state = cv2.cvtColor(tState[:, :, ::-1], cv2.COLOR_BGR2GRAY)
				self.CurrentState = state
				episodeReward = 0
				step = 0
				while not terminated and step < maxSteps:
					if saveVideo and step % self.SaveEveryNFrame == 0:
						action = self.getAction(state=state, onlyExploit=True)
						image = (self.Environment.render('rgb_array')).astype(np.uint8)
						cv2.putText(img=image, text=f"Episode:  {self.CurrentEpisode}", org=(5, 20), fontScale=1.0,
						            thickness=1,
						            color=(0, 225, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
						cv2.putText(img=image, text=f"Reward:  {episodeReward:.3f}", org=(9, 40), fontScale=1.0,
						            thickness=1,
						            color=(0, 225, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
						self.Video.write(image=image)
					else:
						action = self.getAction(state=state)
						self.Environment.render()
					nextState, reward, terminated, _ = self.Environment.step(action)
					if self.UseLeNet:
						nextState = nextState[:, :, ::-1]
					else:
						nextState = cv2.cvtColor(nextState[:, :, ::-1], cv2.COLOR_BGR2GRAY)
					episodeReward += reward
					self.addToMemoryReplay(self.createMemory(state=state, action=action, reward=reward,
					                                         nextState=nextState, isNextStateTerminal=terminated))
					step += 1
					self.CurrentState = nextState
					state = nextState

				tempRewardResults[episode] = episodeReward
				print(f"\t\tTotal Reward: {episodeReward:.4f}")
			self.Avg = np.copy(np.mean(tempRewardResults))
			if self.Avg > self.BestReward:
				self.BestReward = self.Avg
				self.BestWeights = np.asarray(self.ActionValueModel.get_weights(), dtype=object)
			self.RewardsDuringTraining[self.CurrentEpisode // self.TestEveryNEpisodes, :] = tempRewardResults
			pd.DataFrame(self.RewardsDuringTraining).to_csv(
				f"./Data/Results/RewardsDuringTraining_Episode_{self.CurrentEpisode}.csv", header=False)
			print(f"\t\tAverage Reward for Evaluations: {np.mean(tempRewardResults):.4f}")
			print("\t\tFinished Gathering Data\n\n")
			return np.asarray(all_episodes)
		except Exception as _exception:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(f"Exception in ''", _exception)
	
	def printDetails(self):
		print(f"\tCurrent Episode: {self.CurrentEpisode}", end='')
		print(f"\tCurrent Epsilon: {self.Epsilon:.5f}", end='')
		print(f"\tCurrent Episode Reward: {self.TotalRewardForEpisode[self.CurrentEpisode, 0]:.4f}", end='')
		print(f"\tCurrent Average Runtime for Episode: {np.mean(self.RunTimeReplay[self.CurrentEpisode, :]):.4f}",
		      end='\n')
	
	def trainNetwork(self):
		try:
			episodesUntilQMerge = self.StepsToMergeQ
			for episode in range(self.NumEpisodes):
				_startTime = time.time()
				self.CurrentEpisode = episode
				tState = self.Environment.reset()
				if self.UseLeNet:
					state = tState[:, :, ::-1]
				else:
					state = cv2.cvtColor(tState[:, :, ::-1], cv2.COLOR_BGR2GRAY)
				self.CurrentState = state
				step = 0
				episodeReward = 0
				Terminated = False
				while not Terminated:
					step += 1
					action = self.getAction(state=state)
					nextState, reward, Terminated, _ = self.Environment.step(action)
					episodeReward += reward
					if self.UseLeNet:
						nextState = nextState[:, :, ::-1]
					else:
						nextState = cv2.cvtColor(nextState[:, :, ::-1], cv2.COLOR_BGR2GRAY)
					
					# Store Transition
					self.addToMemoryReplay(
						memory=self.createMemory(state=state, action=action, reward=reward, nextState=nextState,
						                         isNextStateTerminal=Terminated))
					state = nextState
					
					if step > self.StepLimit:
						Terminated = True
				
				sampleData = self.sampleFromExperience(sampleSize=self.SampleSize)
				targetY = self.determineTarget(batch=sampleData)
				if self.UseLeNet:
					self.ActionValueModel.fit(
						x=np.reshape(sampleData[:, self.DataIDX["State"]["Start"]:self.DataIDX["State"]["Stop"]],
						             newshape=(self.BatchSize,
						                       self.OriginalDimensions[0],
						                       self.OriginalDimensions[1],
						                       self.OriginalDimensions[2])),
						y=targetY,
						epochs=self.NumEpochs,
						batch_size=self.BatchSize,
						verbose=1)
				else:
					self.ActionValueModel.fit(
						x=sampleData[:, self.DataIDX["State"]["Start"]:self.DataIDX["State"]["Stop"]],
						y=targetY,
						epochs=self.NumEpochs,
						batch_size=self.BatchSize,
						verbose=1)
				episodesUntilQMerge -= 1
				if episodesUntilQMerge <= 0:
					currentModelWeights = np.asarray(self.ActionValueModel.get_weights(), dtype=object)
					targetModelWeights = np.asarray(self.TargetActionValueModel.get_weights(), dtype=object)
					if self.BestWeights is not None:
						new_weights = (self.Alpha * currentModelWeights) + ((1 - self.Alpha) * targetModelWeights) + (self.Alpha * self.BestWeights)
					else:
						new_weights = (self.Alpha * currentModelWeights) + (1 - self.Alpha) * targetModelWeights
					self.TargetActionValueModel.set_weights(new_weights.tolist())
					episodesUntilQMerge = self.StepsToMergeQ
				_endTime = time.time()
				_elapsedTime = _endTime - _startTime
				self.TotalRewardForEpisode[episode, 0] = episodeReward
				self.RunTimeReplay[episode, 0] = _elapsedTime
				if self.DecayEpsilon:
					self.epsilonDecayHelper()
				if episode % self.TestEveryNEpisodes == 0:
					_ = self.getTrainingData(episodes=self.TestSize, saveVideo=self.SaveVideo)
		except Exception as _exception:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(f"Exception in 'trainNetwork'", _exception)


if __name__ == "__main__":
	learningRate = 0.001
	optimizerToUse = "Adam"
	decayRate = 0.001
	numberOfEpisodes = 5001
	batchSize = 1000
	borderlandsAgent = Agent(windowName="Xbox Game Streaming (Test App)",
	                         pathToExe="C:\\Users\\joshu\\OneDrive - Georgia Institute of Technology\\ReinforecementLearningPlayground\\Borderlands\\x360ce\\x360ce.exe",
	                         nnLearningRate=learningRate, epsilon=0.99, gamma=0.99, useTargetNetwork=False,
	                         decayEpsilon=True, epsilonDecay=decayRate, optimizer=optimizerToUse,
	                         numEpisodes=numberOfEpisodes, saveDir="./", batchSize=batchSize,
	                         testing=True, useLeNet=False)
	borderlandsAgent.trainNetwork()
	print()
