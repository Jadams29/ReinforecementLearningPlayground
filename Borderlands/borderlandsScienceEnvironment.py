import time
import os
from windowcapture import WindowCapture
import pyautogui
import cv2
import pytesseract
import numpy as np
import sys
sys.path.append("C:\\Users\\joshu\\OneDrive - Georgia Institute of Technology\\ReinforecementLearningPlayground\\")
from HelperFunctions import LoadX360CE, KeyboardRelay, DataUtil, Network
from Spaces import discrete

pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'
custom_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'


class BorderlandsScienceEnvironment:
	def __init__(self, windowName, testing=False):
		self.windowCapture = WindowCapture(windowName=windowName)
		self.InitialFrame = None
		self.Keyboard = KeyboardRelay.KeyboardRelay()
		self.CurrentFrame = None
		self.CurrentScore = None
		self.PreviousScore = None
		self.CurrentTextAreas = None
		self.TargetTextAreas = None
		self.HighTextAreas = None
		self.TargetScore = None
		self.HighScore = None
		self.CurrentStep = None
		self.MaxSteps = None
		self.Testing = testing
		self.action_space = discrete.Discrete(6)
		
		# self.currentState = None
		# self.currentReward = None
		self.isDone = False
		self.Templates = None
		self.FrameBoundaries = None
		self.Bonus = None
		self.Setup()
	
	def Setup(self):
		if self.InitialFrame is None:
			if self.Testing:
				self.InitialFrame = cv2.blur(cv2.cvtColor(cv2.imread("temp.png"), cv2.COLOR_BGR2GRAY), (3, 3))
				self.CurrentFrame = cv2.blur(cv2.cvtColor(cv2.imread("temp.png"), cv2.COLOR_BGR2GRAY), (3, 3))
			else:
				self.InitialFrame = cv2.blur(self.windowCapture.getNextFrame(), (3, 3))
				self.CurrentFrame = cv2.blur(self.windowCapture.getNextFrame(), (3, 3))
		# self.InitialFrame = self.windowCapture.getNextFrame()
		self.Bonus = False, 100
		self.CurrentTextAreas = np.zeros(shape=(10, 1))
		self.TargetTextAreas = np.zeros(shape=(10, 1))
		self.HighTextAreas = np.zeros(shape=(10, 1))
		self.LoadTemplates()
		self.FrameBoundaries = self.EstablishFrameBoundaries()
		self.PopulateAreaMapping()
		self.CurrentStep = 0
		self.MaxSteps = 100
		return
		
	def PopulateAreaMapping(self):
		for i in range(1, 10):
			currentImage = cv2.imread(f"./Data/Templates/Current_{i}.png", 0)
			currentContours, _ = cv2.findContours(currentImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			currentArea = np.round(cv2.contourArea(currentContours[0]), 1)
			self.CurrentTextAreas[i, 0] = currentArea
			
			targetImage = cv2.imread(f"./Data/Templates/Target_{i}.png", 0)
			targetContours, _ = cv2.findContours(targetImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			targetArea = np.round(cv2.contourArea(targetContours[0]), 1)
			self.TargetTextAreas[i, 0] = targetArea
			
			highImage = cv2.imread(f"./Data/Templates/High_{i}.png", 0)
			highContours, _ = cv2.findContours(highImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			highArea = np.round(cv2.contourArea(highContours[0]), 1)
			self.HighTextAreas[i, 0] = highArea
		print("Templates Loaded")
		return
	
	def EstablishFrameBoundaries(self):
		rowStart = self.Templates["Target Score Template"]["Cords"][0][0] - self.Templates["Target Score Template"][
			"Height"]
		rowEnd = self.InitialFrame.shape[0] - self.Templates["Target Score Template"]["Height"]
		
		colStart = self.Templates["Target Score Template"]["Cords"][0][1] - self.Templates["Target Score Template"][
			"Width"]
		colEnd = self.Templates["High Score Template"]["Cords"][1][1] + self.Templates["Target Score Template"]["Width"]
		return rowStart, rowEnd, colStart, colEnd
	
	def LoadTemplates(self):
		originalDimensions = (1056, 1936)
		# Load Current Score Template
		currentScoreTemplate = cv2.imread("./Data/Templates/currentScore.png", 0)
		
		# Load Target Score Template
		targetScoreTemplate = cv2.imread("./Data/Templates/targetScore.png.", 0)
		
		# Load High Score Template
		highScoreTemplate = cv2.imread("./Data/Templates/highScore.png", 0)
		
		if originalDimensions != self.InitialFrame.shape[:2]:
			xScale = self.InitialFrame.shape[0] / originalDimensions[0]
			yScale = self.InitialFrame.shape[1] / originalDimensions[1]
			currentScoreTemplate = cv2.resize(currentScoreTemplate, None, fx=xScale, fy=yScale,
			                                  interpolation=cv2.INTER_CUBIC)
			targetScoreTemplate = cv2.resize(targetScoreTemplate, None, fx=xScale, fy=yScale,
			                                 interpolation=cv2.INTER_CUBIC)
			highScoreTemplate = cv2.resize(highScoreTemplate, None, fx=xScale, fy=yScale,
			                               interpolation=cv2.INTER_CUBIC)
		
		templates = {"Current Score Template": {"Image": currentScoreTemplate},
		             "Target Score Template": {"Image": targetScoreTemplate},
		             "High Score Template": {"Image": highScoreTemplate}}
		
		for i in ["Current Score Template", "Target Score Template", "High Score Template"]:
			# Store width and height of template in w and h
			w, h = templates[i]["Image"].shape[::-1]
			templates[i]["Width"] = w
			templates[i]["Height"] = h
			# Perform match operations.
			res = cv2.matchTemplate(self.InitialFrame, templates[i]["Image"], cv2.TM_CCOEFF_NORMED)
			
			# Specify a threshold
			threshold = 0.75
			
			# Store the coordinates of matched area in a numpy array
			loc = np.where(res >= threshold)
			pt1 = loc[0][0], loc[1][0]
			pt2 = (pt1[0] + h, pt1[1] + w,)
			templates[i]["Cords"] = pt1, pt2
		
		self.Templates = templates
	
	def areaToScore(self, areas, whichToUse="Current"):
		resultString = ""
		if whichToUse == "Current":
			currentDict = self.CurrentTextAreas
		elif whichToUse == "Target":
			currentDict = self.TargetTextAreas
		elif whichToUse == "High":
			currentDict = self.HighTextAreas
		
		for i in areas:
			resultString += str(np.argmin(np.abs(currentDict - i)))
		
		return int(resultString)
	
	def getCropFromTemplate(self, template):
		pt1, pt2 = template["Cords"]
		return pt2[0], pt2[0] + int(template["Height"] * 2.5), pt1[1], pt2[1]
	
	def thresholdImage(self, image):
		image[image < 70] = 0
		image[image >= 70] = 255
		return image
	
	def getCurrentScore(self):
		try:
			if self.PreviousScore is None:
				self.PreviousScore = self.CurrentScore
			rowStart, rowEnd, columnStart, columnEnd = self.getCropFromTemplate(
				self.Templates["Current Score Template"])
			croppedScore = self.CurrentFrame[rowStart:rowEnd, columnStart:columnEnd]
			croppedScore = self.thresholdImage(croppedScore)
			contours, hierarchy = cv2.findContours(croppedScore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			areas = [np.round(cv2.contourArea(i), 1) for i in contours[::-1]]
			return self.areaToScore(areas, whichToUse="Current")
		except Exception as _exception:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(f"Exception in ''", _exception)
	
	def getTargetScore(self):
		try:
			rowStart, rowEnd, columnStart, columnEnd = self.getCropFromTemplate(self.Templates["Target Score Template"])
			croppedScore = self.CurrentFrame[rowStart:rowEnd, columnStart:columnEnd]
			croppedScore = self.thresholdImage(croppedScore)
			contours, hierarchy = cv2.findContours(croppedScore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			areas = [np.round(cv2.contourArea(i), 1) for i in contours[::-1]]
			return self.areaToScore(areas, whichToUse="Target")
		except Exception as _exception:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(f"Exception in ''", _exception)

	def getHighScore(self):
		try:
			rowStart, rowEnd, columnStart, columnEnd = self.getCropFromTemplate(self.Templates["High Score Template"])
			croppedScore = self.CurrentFrame[rowStart:rowEnd, columnStart:columnEnd]
			croppedScore = self.thresholdImage(croppedScore)
			contours, hierarchy = cv2.findContours(croppedScore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			areas = [np.round(cv2.contourArea(i), 1) for i in contours[::-1]]
			return self.areaToScore(areas, whichToUse="High")
		except Exception as _exception:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
			print(f"Exception in ''", _exception)

	def getState(self):
		self.CurrentScore = self.getCurrentScore()
		self.TargetScore = self.getTargetScore()
		self.HighScore = self.getHighScore()
		
		return self.CurrentFrame[self.FrameBoundaries[0]:self.FrameBoundaries[1], self.FrameBoundaries[2]:self.FrameBoundaries[3]]
	
	def getReward(self):
		return self.CurrentScore - self.PreviousScore
	
	def getProgress(self):
		if self.CurrentScore >= self.HighScore and self.CurrentStep > self.MaxSteps:
			self.Bonus[0] = True
			return True
		elif self.CurrentStep > self.MaxSteps:
			return True
		
	def step(self, action):
		# Execute action
		self.Keyboard.OnlyPressKey(action)
		# Get State
		self.CurrentFrame = self.windowCapture.getNextFrame()
		currentState = self.getState()
		self.isDone = self.getProgress()
		# Get Reward
		currentReward = self.getReward()
		
		if self.Bonus[0]:
			currentReward += self.Bonus[1]
		return currentState, currentReward, self.isDone, "hey"
	
	def reset(self):
		self.Setup()
		if self.Testing:
			return self.getState()
		else:
			if self.CurrentFrame is None:
				self.CurrentFrame = self.windowCapture.getNextFrame()
			return self.getState()
