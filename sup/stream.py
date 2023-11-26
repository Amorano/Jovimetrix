"""
     ██  ██████  ██    ██ ██ ███    ███ ███████ ████████ ██████  ██ ██   ██ 
     ██ ██    ██ ██    ██ ██ ████  ████ ██         ██    ██   ██ ██  ██ ██  
     ██ ██    ██ ██    ██ ██ ██ ████ ██ █████      ██    ██████  ██   ███  
██   ██ ██    ██  ██  ██  ██ ██  ██  ██ ██         ██    ██   ██ ██  ██ ██ 
 █████   ██████    ████   ██ ██      ██ ███████    ██    ██   ██ ██ ██   ██ 

               Procedural & Compositing Image Manipulation Nodes
                    http://www.github.com/amorano/jovimetrix

                    Copyright 2023 Alexander Morano (Joviex)

Test unit for webcam setup.
"""

import cv2
import numpy as np

import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from typing import Any

try:
    from .util import loginfo, logwarn, logerr, gridMake
except:
    from sup.util import loginfo, logwarn, logerr, gridMake

try:
    from .comp import SCALEFIT
except:
    from sup.comp import SCALEFIT

class MediaStream():
    def __init__(self, url:int|str, size:tuple[int, int]=None, fps:float=None, mode:str="NONE", backend:int=None) -> None:
        self.__source = None
        self.__quit = False
        self.__thread = None
        self.__paused = True
        self.__ret = False
        self.__backend = [backend] if backend else [cv2.CAP_ANY]
        self.__frame = np.zeros((1, 1, 3), dtype=np.uint8)
        self.__fps = fps or 60
        self.__zoom = 0
        self.__focus = 0
        self.__exposure = 1

        self.size = size
        self.__mode = mode

        self.__isCam = False
        self.__url = url
        try:
            self.__url = int(url)
            self.__isCam = True
        except ValueError as _:
            pass

        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def __run(self) -> None:
        while not self.__quit:
            waste = time.time() + 1. / self.__fps
            timeout = None
            if not self.__paused:
                newframe = None
                try:
                    self.__ret, newframe = self.__source.read()
                except:
                    self.__ret = False

                if not self.__ret:
                    if not self.__isCam:
                        # for cameras will just ignore; reached the end of the source
                        count = self.__source.get(cv2.CAP_PROP_FRAME_COUNT)
                        pos = self.__source.get(cv2.CAP_PROP_POS_FRAMES)
                        if pos >= count:
                            self.__source.set(cv2.CAP_PROP_POS_FRAMES, 0)

                if newframe is not None:
                    timeout = None
                    self.__frame = SCALEFIT(newframe, self.__size[0], self.__size[1], self.__mode)

                if not self.__ret or newframe is None and timeout is None:
                    timeout = time.time() + 5.

            if timeout is not None and time.time() > timeout:
                self.__source.release()
                logwarn(f"[MediaStream] TIMEOUT ({self.__url})")
                self.capture()

            waste = max(waste - time.time(), 0)
            time.sleep(waste)

        loginfo(f"[MediaStream] STOPPED ({self.__url})")

    def __del__(self) -> None:
        self.release()
        self.__quit = True
        try:
            if self.__thread:
                self.__thread.join()
                del self.__thread
        except AttributeError as _:
            pass
        logwarn(f"[MediaStream] END ({self.__url})")

    def capture(self, size:tuple[int, int]=None, fps:float=60) -> None:
        if self.__source and self.__source.isOpened():
            self.__paused = False
            # logwarn('already captured')
            return

        loginfo(f"[MediaStream] CAPTURE ({self.__url})")
        found = False
        for x in self.__backend:
            self.__source = cv2.VideoCapture(self.__url, x)
            found = self.__source.isOpened()
            if found:
                break

        if not found:
            logerr(f"[MediaStream] CAPTURE FAIL ({self.__url}) ")
            return

        time.sleep(1)
        self.__fps = max(1, self.__fps or self.__source.get(cv2.CAP_PROP_FPS))
        self.__paused = False
        loginfo(f"[MediaStream] CAPTURED ({self.__url})")

    def pause(self) -> None:
        self.__paused = True
        logwarn(f"[MediaStream] PAUSED ({self.__url})")

    def release(self) -> None:
        if self.__source:
            self.__source.release()
            logwarn(f"[MediaStream] RELEASED ({self.__url})")

    @property
    def frame(self) -> tuple[bool, Any]:
        return self.__ret, self.__frame

    @property
    def isOpen(self) -> bool:
        if not self.__source:
            return False
        return self.__source.isOpened()

    @property
    def width(self) -> int:
        return int(self.__source.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.__source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def size(self) -> tuple[int, int]:
        return self.__size

    @size.setter
    def size(self, size:tuple[int, int]) -> None:
        size = size or (32, 32)

        width = int(np.clip(size[0] or 32, 0, 8192))
        height = int(np.clip(size[1] or 32, 0, 8192))

        self.__size = (width, height)
        self.__frame = cv2.resize(self.__frame, self.__size)

    @property
    def mode(self) -> str:
        return self.__mode

    @size.setter
    def mode(self, mode:str) -> None:
        self.__mode = mode

    @property
    def zoom(self) -> float:
        return self.__zoom

    @zoom.setter
    def zoom(self, val: float) -> None:
        self.__zoom = np.clip(val, 0, 1)
        val = 100 + 300 * self.__zoom
        self.__source.set(cv2.CAP_PROP_ZOOM, val)

    @property
    def exposure(self) -> float:
        return self.__exposure

    @exposure.setter
    def exposure(self, val: float) -> None:
        # -10 to -1 range
        self.__exposure = np.clip(val, 0, 1)
        val = -10 + 9 * self.__exposure
        self.__source.set(cv2.CAP_PROP_EXPOSURE, val)

    @property
    def focus(self) -> float:
        return self.__focus

    @focus.setter
    def focus(self, val: float) -> None:
        self.__focus = np.clip(val, 0, 1)
        val = 255 * self.__focus
        self.__source.set(cv2.CAP_PROP_FOCUS, val)

class StreamManager:

    STREAM = {}

    @classmethod
    def devicescan(cls, capture:bool=False) -> None:
        """Indexes all devices that responded and if they are read-only."""

        for stream in StreamManager.STREAM.values():
            if stream:
                del stream
        StreamManager.STREAM = {}

        start = time.time()
        for i in range(5):
            stream = MediaStream(i)
            if capture:
                stream.capture()
                if not stream.isOpen:
                    break
            StreamManager.STREAM[i] = stream

        if capture:
            for stream in StreamManager.STREAM.values():
                stream.release()

        loginfo(f"[StreamManager] SCAN ({time.time()-start:.4})")

    def __init__(self, autoscan=False) -> None:
        StreamManager.devicescan(autoscan)
        loginfo(f"[StreamManager] STREAM {self.streams}")

    def __del__(self) -> None:
        for c in StreamManager.STREAM.values():
            del c

    @property
    def streams(self) -> list[str|int]:
        return list(StreamManager.STREAM.keys())

    @property
    def active(self) -> list[MediaStream]:
        return [stream for stream in StreamManager.STREAM.values() if stream.isOpen]

    def frame(self, url: str) -> tuple[bool, Any]:
        if (stream := StreamManager.STREAM.get(url, None)) is None:
            return False, np.zeros((512, 512, 3), dtype=np.uint8)
        return stream.frame

    def capture(self, url: str, size:tuple[int, int]=None, fps:float=None, backend:int=None) -> MediaStream:
        if (stream := StreamManager.STREAM.get(url, None)) is None:
            stream = StreamManager.STREAM[url] = MediaStream(url=url, size=size, fps=fps, backend=backend)
        stream.capture()
        return stream

    def pause(self, url: str) -> None:
        if (stream := StreamManager.STREAM.get(url, None)) is None:
            stream.pause()

class StreamingHandler(BaseHTTPRequestHandler):
    def __init__(self, outputs, *args, **kwargs) -> None:
        self.__outputs = outputs
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        key = self.path.lower()
        if (data := self.__outputs.get(key, None)) is None:
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        while True:
            try:
                if (frame := data['b']) is not None:
                    _, jpeg = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpeg))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logerr(f"Error: {e}")
                break

class StreamingServer:
    OUT = {}

    @classmethod
    def endpointAdd(cls, name: str, stream: MediaStream) -> None:
        StreamingServer.OUT[name] = {'_': stream, 'b': None}

    def __init__(self, host: str='', port: int=7227) -> None:
        def capture() -> None:
            while True:
                who = list(StreamingServer.OUT.keys())
                for w in who:
                    _, frame = StreamingServer.OUT[w]['_'].frame
                    StreamingServer.OUT[w]['b'] = frame

        capture_thread = threading.Thread(target=capture, daemon=True)
        capture_thread.start()
        self.__host = host
        self.__port = port
        address = (self.__host, self.__port)
        httpd = ThreadingHTTPServer(address, lambda *args: StreamingHandler(StreamingServer.OUT, *args))

        def server() -> None:
            while True:
                httpd.handle_request()

        server_thread = threading.Thread(target=server, daemon=True)
        server_thread.start()
        loginfo("[StreamingServer] STARTED")

def streamReadTest() -> None:
    urls = [
        0,
        1,
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

    streamMGR = StreamManager()

    count = 0
    widthT = 160
    heightT = 120
    chunks = []

    empty = frame = np.zeros((heightT, widthT, 3), dtype=np.uint8)
    while True:
        streams = []
        for x in streamMGR.active:
            ret, chunk = streamMGR.frame(x)
            if not ret or chunk is None:
                streams.append(empty)
                continue
            #chunk = cv2.resize(chunk, (widthT, heightT))
            streams.append(chunk)

        chunks, col, row = gridMake(streams)
        if (countNew := len(streamMGR.active)) != count:
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
            streamMGR.capture(urls[streamIdx % len(urls)], (widthT, heightT))
            streamIdx += 1
        elif val == ord('q'):
            break

    cv2.destroyAllWindows()

def streamWriteTest() -> None:
    ss = StreamingServer()
    sm = StreamManager()

    fpath = 'res/stream-video.mp4'
    #device_video = sm.capture(fpath, fps=30)
    #ss.endpointAdd('/video', device_video)
    device_camera = sm.capture(1)
    ss.endpointAdd('/camera', device_camera)

    device_camera.size = (960, 540)

    modeIdx = 0
    modes = ["NONE", "FIT", "ASPECT", "CROP"]

    _, empty = device_camera.frame
    while 1:
        cv2.imshow("SERVER", empty)
        val = cv2.waitKey(1) & 0xFF
        if val == ord('q'):
            break
        elif val == ord('z'):
            device_camera.exposure -= 0.02
            loginfo(device_camera.exposure)
        elif val == ord('c'):
            device_camera.exposure += 0.02
            loginfo(device_camera.exposure)
        elif val == ord('m'):
            modeIdx = (modeIdx + 1) % len(modes)
            device_camera.mode = modes[modeIdx]
            # device_video.mode = modes[modeIdx]
            loginfo(modes[modeIdx])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    #streamReadTest()
    streamWriteTest()
