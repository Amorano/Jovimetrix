"""Test unit for webcam setup.
"""

import cv2
import numpy as np

import io
import time
import threading
import contextlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from typing import Any

from sup.util import loginfo, logwarn, logerr, gridMake

class WebCamera():
    BROKEN = np.zeros((1, 1, 3), dtype=np.uint8)
    def __init__(self, url:[int|str]=None, width:int=640, height:int=480, fps:float=60, mode:str="FIT") -> None:
        self.__source = None
        self.__running = False
        self.__thread = None
        self.__paused = True

        self.__fps = 0
        self.fps = fps
        self.__width = self.width = width
        self.__height = self.height = height

        self.__ret = False
        self.__frame = np.zeros((width, height, 3), dtype=np.uint8)

        # SCALEFIT
        self.__mode = mode

        self.__backend = [cv2.CAP_ANY]
        # RTSP or CamIndex...
        self.__url = url
        try:
            self.__url = int(url)
        except ValueError as _:
            pass

        f = io.StringIO()
        e = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(e):
            self.__thread = threading.Thread(target=self.__run, daemon=True)
            self.__thread.start()

    def __run(self) -> None:
        def captureStream() -> None:
            found = False
            for x in self.__backend:
                self.__source = cv2.VideoCapture(self.__url, x)
                found = self.__source.isOpened()
                if found:
                    break

            if not found:
                logerr(f"[WebCamera] FAILED ({self.__url}) ")
                self.__running = False
                return

            # time.sleep(0.4)
            self.__paused = False
            self.__running = True
            loginfo(f"[WebCamera] RUNNING ({self.__url})")

        captureStream()

        while self.__running:
            timeout = None
            if not self.__paused:
                self.__ret, self.__frame = self.__source.read()
                if self.__ret and self.__frame is not None:
                    timeout = None
                elif self.__source.isOpened():
                    if timeout is None:
                        timeout = time.time()
                    elif time.time() - timeout > 2:
                        self.__source.release()
                        logwarn(f"[WebCamera] TIMEOUT ({self.__url})")
                        captureStream()

                    self.__frame = WebCamera.BROKEN

                width, height, _ = self.__frame.shape
                if width != self.__width or height != self.__height:
                    self.__frame = cv2.resize(self.__frame, (self.__width, self.__height))

            # waste = (time.time() - waste)
            # waste = min(1, max(0.0001, self.__fps - waste))
            # time.sleep(waste)

        loginfo(f"[WebCamera] STOPPED ({self.__url})")

    def __del__(self) -> None:
        self.release()

    def capture(self) -> None:
        self.__paused = False
        loginfo(f"[WebCamera] CAPTURE ({self.__url})")

    def pause(self) -> None:
        self.__paused = True
        logwarn(f"[WebCamera] PAUSED ({self.__url})")

    def release(self) -> None:
        self.__running = False
        if self.__source:
            self.__source.release()
            logwarn(f"[WebCamera] RELEASED ({self.__url})")

        try:
            if self.__thread:
                del self.__thread
        except AttributeError as _:
            pass
        logwarn(f"[WebCamera] END ({self.__url})")

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
        self.__width = np.clip(width, 0, 8192)

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, height: int) -> None:
        self.__height = np.clip(height, 0, 8192)

    @property
    def frame(self) -> Any:
        return self.__frame

    @property
    def frameResult(self) -> tuple[bool, Any]:
        return self.__ret, self.__frame

    @property
    def opened(self) -> bool:
        if not self.__source:
            return False
        return self.__source.isOpened()

class CameraManager:

    CAMS = {}

    @classmethod
    def devicescan(cls) -> None:
        """Indexes all devices that responded and if they are read-only."""

        CameraManager.CAMS = {}
        start = time.time()
        for i in range(5):
            camera = WebCamera(i)
            camera.capture()
            if not camera.opened:
                return

            frame = None
            timeout = time.time()
            while (frame is None or not ret) and (time.time() - timeout) < 2:
                ret, frame = camera.frameResult

            if ret or frame:
                CameraManager.CAMS[i] = camera

        loginfo(f"[CameraManager] SCAN ({time.time()-start:.4})")

    def __init__(self) -> None:
        CameraManager.devicescan()
        loginfo(f"[CameraManager] CAMS {self.camlist}")

    def __del__(self) -> None:
        for c in CameraManager.CAMS.values():
            c.release()

    @property
    def camlist(self) -> list[WebCamera]:
        return list(CameraManager.CAMS.keys())

    def frame(self, url: str) -> tuple[bool, Any]:
        if (camera := CameraManager.CAMS.get(url, None)) is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return camera.frameResult

    def openURL(self, url: str) -> None:
        if (CameraManager.CAMS.get(url, None)) is None:
            CameraManager.CAMS[url] = WebCamera(url=url)

class StreamingHandler(BaseHTTPRequestHandler):
    def __init__(self, frame_buffer, *args, **kwargs) -> None:
        self.frame_buffer = frame_buffer
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path != '/stream.mjpg':
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while True:
            try:
                frame = self.frame_buffer.get_frame()
                if frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpeg))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
            except Exception as e:
                print(f"Error: {e}")
                break

class FrameBuffer:
    def __init__(self) -> None:
        self.frame = None
        self.lock = threading.Lock()

    def update_frame(self, frame) -> None:
        with self.lock:
            self.frame = frame

    def get_frame(self) -> Any | None:
        with self.lock:
            return self.frame

def streamTester() -> None:
    cap = cv2.VideoCapture(0)
    frame_buffer = FrameBuffer()

    def capture_frames() -> None:
        while True:
            ret, frame = cap.read()
            if ret:
                frame_buffer.update_frame(frame)

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    capture_thread.start()

    try:
        address = ('', 8000)
        httpd = ThreadingHTTPServer(address, lambda *args: StreamingHandler(frame_buffer, *args))
        httpd.serve_forever()
    finally:
        cap.release()

def cameraTester() -> None:
    streams = [
        "rtsp://rtspstream:804359a2ea4669af4edf7feab36ce048@zephyr.rtsp.stream/pattern",

        "http://camera.sissiboo.com:86/mjpg/video.mjpg",
        "http://brandts.mine.nu:84/mjpg/video.mjpg",
        "http://webcam.mchcares.com/mjpg/video.mjpg",
        "http://htadmcam01.larimer.org/mjpg/video.mjpg",
        "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg",
        "https://gbpvcam01.taferresorts.com/mjpg/video.mjpg",

        "http://217.147.30.197:8087/mjpg/video.mjpg",
        "http://173.162.200.86:3123/mjpg/video.mjpg",
        "http://188.113.160.155:47544/mjpg/video.mjpg",
        "http://63.142.183.154:6103/mjpg/video.mjpg",
        "http://104.207.27.126:8080/mjpg/video.mjpg",
        "http://185.133.99.214:8011/mjpg/video.mjpg",
        "http://tapioles.eu:85/mjpg/video.mjpg",
        "http://63.142.190.238:6106/mjpg/video.mjpg",
        "http://77.222.181.11:8080/mjpg/video.mjpg",
        "http://195.196.36.242/mjpg/video.mjpg",
        "http://158.58.130.148/mjpg/video.mjpg",

        "rtsp://rtspstream:a0e429a2f87f5ef980d6a22198ecc1dc@zephyr.rtsp.stream/movie",

        "http://honjin1.miemasu.net/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://clausenrc5.viewnetcam.com:50003/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://tamperehacklab.tunk.org:38001/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://vetter.viewnetcam.com:50000/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard",
    ]
    streamIdx = 0

    camMgr = CameraManager()

    count = 0
    widthT = 160
    heightT = 120
    chunks = []

    empty = frame = np.zeros((heightT, widthT, 3), dtype=np.uint8)
    while True:
        cameras = []
        for x in camMgr.camlist:
            ret, chunk = camMgr.frame(x)
            if not ret or chunk is None:
                cameras.append(empty)
                continue

            chunk = cv2.resize(chunk, (widthT, heightT))
            cameras.append(chunk)

        chunks, col, row = gridMake(cameras)
        if (countNew := len(camMgr.camlist)) != count:
            frame = np.zeros((heightT * row, widthT * col, 3), dtype=np.uint8)
            count = countNew

        i = 0
        for y, strip in enumerate(chunks):
            for x, item in enumerate(strip):
                y1, y2 = y * heightT, (y+1) * heightT
                x1, x2 = x * widthT, (x+1) * widthT
                frame[y1:y2, x1:x2, ] = item
                i += 1

        cv2.imshow("Web Camera", frame)
        val = cv2.waitKey(1) & 0xFF
        if val == ord('c'):
            camMgr.openURL(streams[streamIdx])
            streamIdx += 1
            if streamIdx >= len(streams):
                streamIdx = 0
        elif val == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    streamTester()
    # cameraTester()
