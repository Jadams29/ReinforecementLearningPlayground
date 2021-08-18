import copy

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
from mss import mss
import winGuiAuto
from PIL import ImageGrab, Image


class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, windowName):
        self.entireFrame = None
        self.croppedFrame = None
        self.capture = mss()
        self.frameWidthStartOffset = 200
        self.hwnd = win32gui.FindWindow(None, windowName)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(windowName))
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.window_cords = copy.copy(window_rect)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def getNextFrame(self):
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        w = right - left
        h = bot - top

        win32gui.SetForegroundWindow(self.hwnd)
        # time.sleep(1.0)

        hdesktop = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hdesktop)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (w, h), mfcDC, (left, top), win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        image = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)
        img = np.array(image)
        img = img[:, :, ::-1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.entireFrame = np.copy(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def list_window_names(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    def get_screen_position(self, pos):
        return pos[0] + self.offset_x, pos[1] + self.offset_y