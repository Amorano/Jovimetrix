"""Test unit for webcam setup.
"""

import cv2
import numpy as np

import time
import threading
import concurrent.futures
from typing import Any

loginfo = print

class WebCamera:
    def __init__(self, url:[int|str]=None, width:int=640, height:int=480, fps:float=30) -> None:
        self.__camera = None
        self.__thread = None
        self.__running = False
        self.__paused = False
        self.__url = url
        self.__ret = False
        self.__frame = np.zeros((width, height, 3), dtype=np.uint8)

        found = False
        try:
            url = int(url)
            for x in [cv2.CAP_DSHOW, cv2.CAP_V4L, cv2.CAP_ANY]:
                self.__camera = cv2.VideoCapture(url, x)
                if self.__camera.isOpened():
                    found = True
                    break

        except ValueError as _:
            self.__camera = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            found = self.__camera.isOpened()

        if not found:
            raise Exception(f"[WebCamera] FAILED ({url}) ")

        self.__camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.__size = (width, height)
        self.__width = width
        self.__height = height
        self.width = width
        self.height = height
        self.fps = fps
        loginfo(f"[WebCamera] INIT ({url}) ")

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
                loginfo(f"[WebCamera] RELEASED ({self.__url})")

    def capture(self) -> None:
        self.__paused = False
        self.__running = self.__camera.isOpened()
        self.__thread = threading.Thread(target=self.__process, daemon=True)
        self.__thread.start()
        # give it a beat
        time.sleep(0.4)
        loginfo(f"[WebCamera] CAPTURE ({self.__url})")

    def stop(self) -> None:
        self.__paused = True
        self.__running = False
        if self.__thread:
            self.__thread.join()
        self.__thread = None
        loginfo(f"[WebCamera] STOPPED ({self.__url})")

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

    CAMS = {}

    @classmethod
    def check_cam_async(cls, idx: int) -> None:
        try:
            camera = WebCamera(idx)
        except:
            return

        camera.capture()
        if not camera.opened:
            return
        frame = None
        while frame is None or not ret:
            ret, frame = camera.frameResult
        camera.stop()
        CameraManager.CAMS[idx] = camera

    @classmethod
    def devicescan(cls) -> None:
        """Indexes all devices that responded and if they are read-only."""

        CameraManager.CAMS = {}
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(cls.check_cam_async, i) for i in range(4)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, camera = future.result()
                    CameraManager.CAMS[idx] = camera
                except TypeError as _:
                    pass

        loginfo(f"[CameraManager] INIT ({time.time()-start:.4})")

    def __init__(self) -> None:
        CameraManager.devicescan()
        loginfo(f"[CameraManager] CAMS {self.camlist}")

    def __del__(self) -> None:
        for c in CameraManager.CAMS.values():
            c.release()

    @property
    def camlist(self) -> list[WebCamera]:
        return list(CameraManager.CAMS.keys())

    def frame(self, idx: int) -> Any:
        if (camera := CameraManager.CAMS.get(idx, None)) is None:
            return None
        return camera.frame

    def capture(self, idx: int) -> None:
        if (camera := CameraManager.CAMS.get(idx, None)) is None:
            return
        camera.capture()

    def captureAll(self) -> None:
        for camera in CameraManager.CAMS.values():
            camera.capture()

    def addURL(self, url: str) -> Any:
        return WebCamera(url=url)

if __name__ == "__main__":
    camMgr = CameraManager()
    camMgr.addURL("rtsp://rtspstream:804359a2ea4669af4edf7feab36ce048@zephyr.rtsp.stream/pattern")
    camMgr.captureAll()

    width = 1280
    height = 384
    width2 = width // 3
    height2 = height #// 2

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    while True:
        for x in camMgr.camlist:
            if (chunk := camMgr.frame(x)) is not None:
                chunk = cv2.resize(chunk, (width2, height))
                col = x % 2
                frame[: height: , col * width2: (col+1) * width2: , ] = chunk

        cv2.imshow("Web Camera", frame)
        val = cv2.waitKey(1) & 0xFF
        if val == ord('q'):
            break
        elif val == ord('k'):
            camMgr.addURL("rtsp://rtspstream:804359a2ea4669af4edf7feab36ce048@zephyr.rtsp.stream/pattern")

    cv2.destroyAllWindows()
