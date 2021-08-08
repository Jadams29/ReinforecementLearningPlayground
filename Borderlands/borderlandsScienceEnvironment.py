import time

from windowcapture import WindowCapture
import pyautogui
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C://Program Files//Tesseract-OCR//tesseract.exe'


class BorderlandsScienceEnvironment:
    def __init__(self, windowName):
        self.windowCapture = WindowCapture(windowName=windowName)
        self.currentState = None
        self.currentReward = None
        self.isDone = False
        self.nA = 5  # left, down, right, up, A
        self.actions = {"Left": (2141, 628),
                        "Down": (2158, 647),
                        "Right": (2177, 629),
                        "Up": (2159, 610),
                        "A": (3694, 571)}
        self.rewardCords = {"Current Score": {"X": 103, "Y": 920, "Width": 112, "Height": 45},
                            "High Score": {"X": 103, "Y": 1112, "Width": 107, "Height": 33},
                            "Target Score": {"X": 103, "Y": 708, "Width": 131, "Height": 33}}

        pass
    
    def getAllActions(self):
        return self.actions
    
    
    # def getProgress(self):
    #     # Load the progress bar, estimate % of complete
    #     buffer = 10
    #     currentScoreImg = self.currentState[self.rewardCords["Current Score"]["X"] - buffer:self.rewardCords["Current Score"]["X"] + self.rewardCords["Current Score"]["Height"] + buffer,
    #                       self.rewardCords["Current Score"]["Y"]- buffer: self.rewardCords["Current Score"]["Y"] + self.rewardCords["Current Score"]["Width"] + buffer]
    #     # currentScoreImg = cv2.cvtColor(currentScoreImg, cv2.COLOR_GRAY2BGR)
    #     currentScoreImg = cv2.rectangle(currentScoreImg, pt1=(0, 0), pt2=(self.rewardCords["Current Score"]["Width"], self.rewardCords["Current Score"]["Height"]),
    #                   color=255, thickness=4)
    #     currentScoreImg -= 255
    #     cv2.imwrite("out.png", currentScoreImg)
    #
    #     targetScoreImg = self.currentState[self.rewardCords["Target Score"]["X"]:self.rewardCords["Target Score"]["X"] + self.rewardCords["Target Score"]["Height"],
    #                       self.rewardCords["Target Score"]["Y"]: self.rewardCords["Target Score"]["Y"] + self.rewardCords["Target Score"]["Width"]]
    #
    #     highScoreImg = self.currentState[self.rewardCords["High Score"]["X"]:self.rewardCords["High Score"]["X"] + self.rewardCords["High Score"]["Height"],
    #                       self.rewardCords["High Score"]["Y"]: self.rewardCords["High Score"]["Y"] + self.rewardCords["High Score"]["Width"]]
    #     txt = pytesseract.image_to_string(currentScoreImg)
    #     return 1
        
    def getReward(self):
        #
        return 1
    
    def processAction(self, action):
        pyautogui.leftClick(self.actions[action])
    
    def step(self, action):
        # Execute action
        time.sleep(1)
        for tAction in ["Right", "Right", "Up", "Up", "Right", "Right", "Left", "Down", "A"]:
            self.processAction(action=tAction)
            time.sleep(1.0)
            
        self.currentState = self.windowCapture.getNextFrame()
        
        self.isDone = self.getProgress()
        # Get Reward
        self.currentReward = self.getReward()
        return self.currentState, self.currentReward, self.isDone
        
    def reset(self):
        if self.currentState is not None:
            self.currentReward = 0
            return self.currentState
        else:
            self.currentReward = 0
            return self.windowCapture.getNextFrame()
