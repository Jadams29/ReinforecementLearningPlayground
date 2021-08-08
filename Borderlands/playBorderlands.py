import time

import cv2
import pyautogui
import numpy as np

from borderlandsScienceEnvironment import BorderlandsScienceEnvironment


class Agent:
	def __init__(self, windowName):
		self.Environment = BorderlandsScienceEnvironment(windowName=windowName)
		pass


if __name__ == "__main__":
	borderlandsAgent = Agent(windowName="Xbox Game Streaming (Test App)")
	state = borderlandsAgent.Environment.reset()
	borderlandsAgent.Environment.step("Up")
	print()
	# wind = windowcapture.WindowCapture(window_name="Xbox Game Streaming (Test App)")
	# wind.list_window_names()
	# img_in = wind.getNextFrame()
	# cv2.imshow("img", img_in)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# print()
