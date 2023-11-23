"""Test unit for webcam setup.
"""

import cv2
import numpy as np

import time
import threading
from typing import Any

loginfo = print

class WebCamera:

    def __init__(self, idx:int=0, width:int=640, height:int=480, fps:float=30) -> None:
        self.__idx = idx
        self.__ret = False
        self.__frame = None
        self.__paused = False
        self.__running = False
        self.__thread = None
        self.__camera = None

        for x in [cv2.CAP_DSHOW, cv2.CAP_V4L, cv2.CAP_ANY]:
            self.__camera = cv2.VideoCapture(idx, x)
            if not self.__camera.isOpened():
                raise Exception(f"CAMERA ({idx}) FAILED")
            break

        self.__camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.__size = (width, height)
        self.__width = width
        self.__height = height
        self.width = width
        self.height = height
        self.fps = fps
        loginfo(f"[WebCamera] CAMERA ({idx}) INIT")

    def __process(self) -> None:
        while self.__running:
            waste = time.time()
            if not self.__paused:
                self.__ret, self.__frame = self.__camera.read()
                if self.__ret and self.__frame is not None:
                    if self.__size[0] != self.__width or self.__size[1] != self.__height:
                        self.__frame = cv2.resize(self.__frame, self.__size)

            waste = (time.time() - waste)
            waste = min(1, max(0.001, self.__fps - waste))
            time.sleep(waste)

    def __clearCamera(self) -> None:
        self.stop()
        self.release()

    def __del__(self) -> None:
        self.__clearCamera()

    def release(self) -> None:
        if self.__camera:
            if self.__camera.isOpened():
                self.__camera.release()
                loginfo(f"[WebCamera] CAMERA ({self.__idx}) RELEASED")

    def capture(self) -> None:
        self.__paused = False
        self.__running = self.__camera.isOpened()
        self.__thread = threading.Thread(target=self.__process, daemon=True)
        self.__thread.start()
        # give it a beat
        time.sleep(0.5)
        loginfo(f"[WebCamera] CAMERA ({self.__idx}) CAPTURE")

    def stop(self) -> None:
        self.__paused = True
        self.__running = False
        if self.__thread:
            self.__thread.join()
        self.__thread = None
        loginfo(f"[WebCamera] CAMERA ({self.__idx}) STOPPED")

    @property
    def paused(self) -> bool:
        return self.__paused

    @paused.setter
    def paused(self, pause: bool) -> None:
        self.__paused = pause

    @property
    def fps(self) -> float:
        return 1. / self.__fps

    @fps.setter
    def fps(self, fps: float) -> None:
        self.__fps = 1. / max(1, fps)

    @property
    def width(self) -> int:
        return self.__width

    @width.setter
    def width(self, width: int) -> None:
        if self.__camera:
            self.__camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.__width = int(self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.__size = (width, self.__height)

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, height: int) -> None:
        if self.__camera:
            self.__camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.__height = int(self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.__size = (self.__width, height)

    @property
    def frame(self) -> Any:
        return self.__frame

    @property
    def frameResult(self) -> tuple[bool, Any]:
        return self.__ret, self.__frame

    @property
    def opened(self) -> bool:
        if not self.__camera:
            return False
        return self.__camera.isOpened()

class CameraManager:
    @classmethod
    def devicescan(cls) -> dict[int, WebCamera]:
        """Indexes all devices that responded and if they are read-only."""

        idx = 0
        device = {}
        start = time.time()

        while True:
            try:
                camera = WebCamera(idx)
            except:
                break

            camera.capture()
            if not camera.opened:
                break

            ret, _ = camera.frameResult
            camera.stop()
            if ret:
                device[idx] = camera
            idx += 1

        loginfo(f"[CameraManager] INIT ({time.time()-start:.4})")
        return device

    def __init__(self) -> None:
        # scan and link in all cameras?
        self.__cams = CameraManager.devicescan()
        loginfo(f"[CameraManager] CAMS {list(self.__cams.keys())}")

    def __del__(self) -> None:
        for c in self.__cams.values():
            if c:
                del c

    @property
    def camlist(self) -> list[WebCamera]:
        return list(self.__cams.keys())

    def frame(self, idx: int) -> Any:
        if (camera := self.__cams.get(idx, None)) is None:
            return None
        return camera.frame

    def capture(self, idx: int) -> None:
        if (camera := self.__cams.get(idx, None)) is None:
            return
        camera.capture()

if __name__ == "__main__":
    camMgr = CameraManager()
    camMgr.capture(0)

    width = 1024
    height = 384
    width2 = width // 2
    height2 = height #// 2

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    while True:
        for x in range(2):
            if (chunk := camMgr.frame(x)) is not None:
                chunk = cv2.resize(chunk, (width2, height))
                col = x % 2
                frame[: height: , col * width2: (col+1) * width2: , ] = chunk

        cv2.imshow("Web Camera", frame)
        val = cv2.waitKey(1) & 0xFF
        if val == ord('q'):
            break

    cv2.destroyAllWindows()
