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

import json
import os
import time
import threading
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

try:
    from . import util
    from .util import loginfo, logwarn, logerr, logdebug, gridMake
    from .comp import geo_scalefit
except:
    import util
    from sup.util import loginfo, logwarn, logerr, logdebug, gridMake
    from sup.comp import geo_scalefit

# =============================================================================

class StreamMissingException(Exception): pass

# =============================================================================
# === MEDIA ===
# =============================================================================

class MediaStreamBase:

    TIMEOUT = 5.

    def __init__(self, url:int|str, size:tuple[int, int]=None, fps:float=None, mode:str="NONE", callback=None) -> None:
        self.__quit = False
        self.__paused = False
        self.__captured = False
        self.__ret = False
        self.__width = self.__height = 64
        self.__frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.size(size[0], size[1])
        self.__fps = fps or 60
        self.__mode = mode
        self.__url = url
        self.__timeout = None
        self.__callback = callback

        try:
            self.__url = int(url)
        except:
            pass

        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def run(self) -> tuple[bool, Any]:
        return True, self.__frame

    def capture(self) -> None:
        return True

    def __run(self) -> None:
        while not self.__quit:

            waste = time.time() + 1. / self.__fps

            if not self.__paused:
                if not self.__captured:
                    pause = self.__paused
                    self.__paused = True

                    if not self.capture():
                        logerr(f"[MediaStream] CAPTURE FAIL ({self.__url}) ")
                        self.__quit = True
                        break

                    self.__paused = pause
                    self.__captured = True

                    if self.__callback:
                        self.__callback[0](self, *self.__callback[1:])
                    loginfo(f"[MediaStream] CAPTURED ({self.url})")

                self.__ret, newframe = self.run()
                if newframe is not None:
                    self.__timeout = None
                    self.__frame = geo_scalefit(newframe, self.__width, self.__height, self.__mode)

                if self.__timeout is None and (not self.__ret or newframe is None):
                    self.__timeout = time.time() + self.TIMEOUT

            if self.__timeout is not None and time.time() > self.__timeout:
                self.__timeout = None
                self.__quit = True
                logwarn(f"[MediaStream] TIMEOUT ({self.__url})")

            waste = max(waste - time.time(), 0)
            time.sleep(waste)

        loginfo(f"[MediaStream] STOPPED ({self.__url})")
        self.end()

    def end(self) -> None:
        self.release()
        self.__quit = True
        logwarn(f"[MediaStream] END ({self.__url})")

    def __del__(self) -> None:
        self.end()

    def release(self) -> None:
        self.__captured = False
        logwarn(f"[MediaStream] RELEASED ({self.url})")

    def play(self) -> None:
        self.__paused = False

    def pause(self) -> None:
        self.__paused = True
        logwarn(f"[MediaStream] PAUSED ({self.url})")

    def size(self, width, height) -> None:
        self.__width = max(64, width)
        self.__height = max(64, height)
        self.__frame = cv2.resize(self.__frame, (self.__width, self.__height))
        logdebug(f"[MediaStream] SIZE ({self.__width}, {self.__height})")

    @property
    def captured(self) -> bool:
        return self.__captured

    @property
    def url(self) -> str:
        return self.__url

    @property
    def frame(self) -> tuple[bool, Any]:
        return self.__ret, self.__frame

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def mode(self) -> str:
        return self.__mode

    @mode.setter
    def mode(self, mode:str) -> None:
        self.__mode = mode

    @property
    def fps(self) -> float:
        return self.__fps

    @fps.setter
    def fps(self, val: float) -> None:
        self.__fps = max(1, val)

class MediaStreamDevice(MediaStreamBase):
    def __init__(self, url:int|str, size:tuple[int, int]=None, fps:float=None, mode:str="NONE", backend:int=None, callback=None) -> None:

        self.__source = None
        self.__backend = [backend] if backend else [cv2.CAP_V4L, cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        self.__zoom = 0
        self.__focus = 0
        self.__exposure = 1

        super().__init__(url, size, fps, mode, callback=callback)

    def run(self) -> tuple[bool, Any]:
        if self.__source is None:
            return False, None

        ret = False
        newframe = None
        try:
            ret, newframe = self.__source.read()
        except:
            pass

        if not ret and self.__source:
            # for cameras will just ignore; reached the end of the source
            count = self.__source.get(cv2.CAP_PROP_FRAME_COUNT)
            pos = self.__source.get(cv2.CAP_PROP_POS_FRAMES)
            if pos >= count:
                self.__source.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return ret, newframe

    def capture(self) -> None:
        timeout = time.time() + self.TIMEOUT

        while self.__source is None and time.time() < timeout:
            for x in self.__backend:
                logdebug(f"[MediaStreamDevice] CAPTURE BACK {x}")
                self.__source = cv2.VideoCapture(self.url, x)
                time.sleep(0.5)
                if self.captured:
                    break
                self.__source = None

            if self.__source is None:
                return False

        width = int(self.__source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.__source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size(width, height)
        # self.fps = self.__source.get(cv2.CAP_PROP_FPS)
        return True

    def end(self) -> None:
        super().end()
        try:
            if self.__thread:
                self.__thread.join()
                del self.__thread
        except AttributeError as _:
            pass

    @property
    def captured(self) -> bool:
        if self.__source is None:
            return False
        return self.__source.isOpened()

    def release(self) -> None:
        if self.__source:
            self.__source.release()
        super().release()

    @property
    def zoom(self) -> float:
        return self.__zoom

    @zoom.setter
    def zoom(self, val: float) -> None:
        self.__zoom = np.clip(val, 0, 1)
        val = 100 + 300 * self.__zoom
        self.__source.set(cv2.CAP_PROP_ZOOM, val)
        logdebug(f"[MediaStream] ZOOM ({self.__zoom})")

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

class MediaStreamComfyUI(MediaStreamBase):
    """A stream coming from a comfyui node."""

    def run(self, image: cv2.Mat) -> tuple[bool, cv2.Mat]:
        return True, geo_scalefit(image, self.__width, self.__height, self.__mode)

    def capture(self) -> bool:
        return True

class StreamManager:

    STREAM = {}

    @classmethod
    def devicescan(cls) -> None:
        """Indexes all devices that responded and if they are read-only."""

        def callback(stream, *arg) -> None:
            i = arg[0]
            StreamManager.STREAM[i] = stream

        for stream in StreamManager.STREAM.values():
            if stream:
                del stream
        StreamManager.STREAM = {}

        start = time.time()

        for i in range(5):
            stream = MediaStreamDevice(i, callback=(callback, i,) )

        loginfo(f"[StreamManager] SCAN ({time.time()-start:.4})")

    def __init__(self) -> None:
        #StreamManager.devicescan()
        loginfo(f"[StreamManager] STREAM {self.streams}")

    def __del__(self) -> None:
        if StreamManager:
            for c in StreamManager.STREAM.values():
                del c

    @property
    def streams(self) -> list[str|int]:
        return list(StreamManager.STREAM.keys())

    @property
    def active(self) -> list[MediaStreamDevice]:
        return [stream for stream in StreamManager.STREAM.values() if stream.captured]

    def frame(self, url: str) -> tuple[bool, Any]:
        if (stream := StreamManager.STREAM.get(url, None)) is None:
            # attempt to capture first time...
            stream = self.capture(url)

        if not stream.captured:
            stream.capture()

        return stream.frame

    def capture(self, url: str, size:tuple[int, int]=None, fps:float=None, backend:int=None,
                static:bool=False, callback:object=None) -> MediaStreamBase:

        if (stream := StreamManager.STREAM.get(url, None)) is None:
            if static:
                stream = StreamManager.STREAM[url] = MediaStreamComfyUI(url, size, fps, callback=callback)
                logdebug(f"[StreamManager] MediaStreamComfyUI")
            else:
                stream = StreamManager.STREAM[url] = MediaStreamDevice(url, size, fps, backend=backend, callback=callback)
                logdebug(f"[StreamManager] MediaStream")
        stream.capture()
        return stream

    def pause(self, url: str) -> None:
        if (stream := StreamManager.STREAM.get(url, None)) is None:
            return
        stream.pause()

class StreamingHandler(BaseHTTPRequestHandler):
    def __init__(self, outputs, *args, **kwargs) -> None:
        self.__outputs = outputs
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        key = self.path.lower()

        # Check if the key exists in your data dictionary
        if key in self.__outputs:
            data = self.__outputs[key]

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

        elif key == 'jovimetrix':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            data = json.dumps(data)
            self.wfile.write(data.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

class StreamingServer:
    OUT = {}

    @classmethod
    def endpointAdd(cls, name: str, stream: MediaStreamDevice) -> None:
        StreamingServer.OUT[name] = {'_': stream, 'b': None}
        logdebug(f"[StreamingServer] ENDPOINT_ADD ({name})")

    def __init__(self, host: str='', port: int=7227) -> None:
        self.__host = host
        self.__port = port
        self.__address = (self.__host, self.__port)
        self.__thread_server = threading.Thread(target=self.__server, daemon=True)
        self.__thread_server.start()
        self.__thread_capture = threading.Thread(target=self.__capture, daemon=True)
        self.__thread_capture.start()
        loginfo("[StreamingServer] STARTED")

    def __server(self) -> None:
        httpd = ThreadingHTTPServer(self.__address, lambda *args: StreamingHandler(StreamingServer.OUT, *args))
        while True:
            httpd.handle_request()

    def __capture(self) -> None:
        while True:
            current = StreamingServer.OUT.copy()
            for k, v in current.items():
                if (device := v['_']) is not None:
                    _, frame = device.frame
                    StreamingServer.OUT[k]['b'] = frame

# =============================================================================
# === MEDIA ===
# =============================================================================

def gridImage(data: list[object], width: int, height: int) -> np.ndarray:
    #@TODO: makes poor assumption all images are the same dimensions.
    chunks, col, row = gridMake(data)
    frame = np.zeros((height * row, width * col, 3), dtype=np.uint8)
    i = 0
    for y, strip in enumerate(chunks):
        for x, item in enumerate(strip):
            y1, y2 = y * height, (y+1) * height
            x1, x2 = x * width, (x+1) * width
            frame[y1:y2, x1:x2, ] = item
            i += 1

    return frame

# =============================================================================
# === GLOBAL CONFIG ===
# =============================================================================

# auto-scan the camera ports on startup?
STREAMAUTOSCAN = os.getenv("JOV_STREAM_AUTO", '').lower() in ('true', '1', 't')
STREAMMANAGER = StreamManager()

STREAMSERVER:StreamingServer = None
if (val := os.getenv("JOV_STREAM_SERVER", '').lower() in ('true', '1', 't')):
    STREAMSERVER = StreamingServer()

STREAMHOST = os.getenv("JOV_STREAM_HOST", '')
STREAMPORT = 7227
try: STREAMPORT = int(os.getenv("JOV_STREAM_PORT", STREAMPORT))
except: pass

# =============================================================================
# === TESTING ===
# =============================================================================

def streamReadTest() -> None:
    urls = [
        1,
        0,
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

        "http://honjin1.miemasu.net/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://clausenrc5.viewnetcam.com:50003/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://tamperehacklab.tunk.org:38001/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://vetter.viewnetcam.com:50000/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard",
    ]
    streamIdx = 0

    widthT = 480
    heightT = 320

    empty = np.zeros((heightT, widthT, 3), dtype=np.uint8)
    STREAMMANAGER.capture(urls[streamIdx % len(urls)], (widthT, heightT))
    streamIdx += 1

    while True:
        streams = []
        for x in STREAMMANAGER.active:
            ret, chunk = x.frame
            if not ret or chunk is None:
                e = np.zeros((heightT, widthT, 3), dtype=np.uint8)
                streams.append(e)
                continue

            chunk = cv2.resize(chunk, (widthT, heightT))
            streams.append(chunk)

        if len(streams) > 0:
            frame = gridImage(streams, widthT, heightT)
        else:
            frame = empty

        cv2.imshow("Media", frame)
        val = cv2.waitKey(1) & 0xFF
        if val == ord('c'):
            STREAMMANAGER.capture(urls[streamIdx % len(urls)], (widthT, heightT))
            streamIdx += 1
        elif val == ord('q'):
            break

    cv2.destroyAllWindows()

def streamWriteTest() -> None:
    # print(cv2.getBuildInformation())

    ss = StreamingServer()

    device = STREAMMANAGER.capture(0)
    ss.endpointAdd(f'/stream/0', device)

    device = STREAMMANAGER.capture(1)
    ss.endpointAdd(f'/stream/1', device)

    fpath = 'res/stream-video.mp4'
    device = STREAMMANAGER.capture(fpath)
    ss.endpointAdd('/media', device)
    device.size = (960, 540)
    device.fps = 30

    modeIdx = 0
    modes = ["NONE", "FIT", "ASPECT", "CROP"]

    empty = np.zeros((256, 256, 3), dtype=np.uint8)
    while 1:
        cv2.imshow("SERVER", empty)
        val = cv2.waitKey(1) & 0xFF
        if val == ord('q'):
            break
        elif val == ord('m'):
            modeIdx = (modeIdx + 1) % len(modes)
            for x in STREAMMANAGER.active:
                x.mode = modes[modeIdx]
            loginfo(modes[modeIdx])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    streamReadTest()
    # streamWriteTest()
