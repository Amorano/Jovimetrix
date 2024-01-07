"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Media Stream Support
"""

import os
import json
import time
import threading
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from loguru import logger

from Jovimetrix import Singleton, MIN_IMAGE_SIZE
from Jovimetrix.sup.image import image_grid, image_load

# =============================================================================

class StreamMissingException(Exception): pass

# =============================================================================
# === GLOBAL CONFIG ===
# =============================================================================

STREAMHOST = os.getenv("JOV_STREAM_HOST", '')
STREAMPORT = 7227
try: STREAMPORT = int(os.getenv("JOV_STREAM_PORT", STREAMPORT))
except: pass

# =============================================================================
# === MEDIA ===
# =============================================================================

def camera_list() -> list:
    camera_list = []
    failed = 0
    idx = 0
    while failed < 2:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            camera_list.append(idx)
            cap.release()
        else:
            failed += 1
        idx += 1
    return camera_list

class MediaStreamBase:

    TIMEOUT = 5.

    def __init__(self, url:int|str, callback:object, fps:float=30) -> None:
        self.__quit = False
        self.__paused = False
        self.__captured = False
        self.__ret = False
        self.__fps = fps
        self.__url = url
        try: self.__url = int(url)
        except: pass
        self.__size = (0, 0)
        self.__timeout = None
        self.__callback = callback
        self.__frame = None
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def __run(self) -> None:
        while not self.__quit:

            delta = 1. / self.__fps
            waste = time.perf_counter() + delta

            if not self.__paused:
                if not self.__captured:
                    pause = self.__paused
                    self.__paused = True

                    if not self.capture():
                        logger.error(f"CAPTURE FAIL [{self.__url}]")
                        self.__quit = True
                        break

                    self.__paused = pause
                    self.__captured = True
                    logger.info(f"CAPTURED [{self.__url}]")

                # call the run capture frame command on subclasses
                newframe = None
                if self.__callback is not None:
                    self.__ret, newframe = self.__callback()
                    if newframe is not None:
                        self.__frame = newframe
                        self.__timeout = None

                if self.__timeout is None and (not self.__ret or newframe is None):
                    self.__timeout = time.perf_counter() + self.TIMEOUT

            if self.__timeout is not None and time.perf_counter() > self.__timeout:
                self.__timeout = None
                self.__quit = True
                logger.warning(f"TIMEOUT [{self.__url}]")

            waste = max(waste - time.perf_counter(), delta)
            time.sleep(waste)

        logger.info(f"STOPPED [{self.__url}]")
        self.end()

    def __del__(self) -> None:
        self.end()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def capture(self) -> None:
        self.__captured = True
        return self.__captured

    def end(self) -> None:
        self.release()
        self.__quit = True

    def release(self) -> None:
        self.__captured = False

    def play(self) -> None:
        self.__paused = False

    def pause(self) -> None:
        self.__paused = True

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
    def fps(self) -> float:
        return self.__fps

    @fps.setter
    def fps(self, val: float) -> None:
        self.__fps = max(1, val)

    @property
    def size(self) -> tuple[int, int]:
        return self.__size

    @size.setter
    def size(self, val: tuple[int, int]) -> None:
        self.__size = val

class MediaStreamURL(MediaStreamBase):
    """A media point (could be a camera index)."""
    def __init__(self, url:int|str, fps:float=30) -> None:
        self.__single = None
        self.__source = None
        super().__init__(url, self.__callback, fps)

    def __callback(self) -> tuple[bool, Any]:
        if self.__source is None:
            return False, None

        ret = False
        try:
            ret, result = self.__source.read()
        except:
            pass

        if self.__single is not None:
            result = self.__single
            ret = True
        else:
            if result is not None:
                self.__single = result
            elif not ret:
                count = self.__source.get(cv2.CAP_PROP_FRAME_COUNT)
                pos = self.__source.get(cv2.CAP_PROP_POS_FRAMES)
                if pos >= count:
                    self.__source.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return ret, result

    def capture(self) -> bool:
        if self.captured:
            return True
        self.__source = cv2.VideoCapture(self.url, cv2.CAP_ANY)
        if self.captured:
            return True
        return False

    @property
    def captured(self) -> bool:
        if self.__source is None:
            return False
        return self.__source.isOpened()

    def release(self) -> None:
        if hasattr(self, "__source") and self.__source is not None:
            self.__source.release()
        super().release()

class MediaStreamDevice(MediaStreamBase):
    """A system device like a web camera."""
    def __init__(self, url:int|str, fps:float=30) -> None:
        self.__focus = 0
        self.__exposure = 1
        self.__zoom = 0
        self.__source = None
        super().__init__(url, self.__callback, fps)

    def __callback(self) -> tuple[bool, Any]:
        try:
            return self.__source.read()
        except:
            pass
        return False, None

    def capture(self) -> bool:
        if self.captured:
            return True
        self.__source = cv2.VideoCapture(self.url, cv2.CAP_ANY)
        if self.captured:
            time.sleep(0.2)
            width = self.__source.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.__source.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.size = (width, height)
            return True
        return False

    @property
    def captured(self) -> bool:
        if self.__source is None:
            return False
        return self.__source.isOpened()

    def release(self) -> None:
        if hasattr(self, "__source") and self.__source is not None:
            self.__source.release()
        super().release()

    @MediaStreamBase.size.setter
    def size(self, val) -> None:
        width = max(MIN_IMAGE_SIZE, val[0] if val is not None else MIN_IMAGE_SIZE)
        height = max(MIN_IMAGE_SIZE, val[1] if val is not None else MIN_IMAGE_SIZE)
        self.__source.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__source.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

class MediaStreamStatic(MediaStreamBase):
    """A stream coming from ComfyUI."""
    def __init__(self, url:str) -> None:
        self.image = None
        super().__init__(url, self.__callback)

    def __callback(self) -> tuple[bool, Any]:
        return True, self.image

class MediaStreamFile(MediaStreamBase):
    """A file served from a local file using file:// as the 'uri'."""
    def __init__(self, url:str) -> None:
        self.__image = image_load(url)[0]
        super().__init__(url, self.__callback)

    def __callback(self) -> tuple[bool, Any]:
        return True, self.__image

class StreamManager(metaclass=Singleton):
    STREAM = {}
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
        return stream.frame

    def capture(self, url: str, fps:float=30, static:bool=False, endpoint:str=None) -> MediaStreamBase:
        if (stream := StreamManager.STREAM.get(url, None)) is None:
            try:
                if static:
                    StreamManager.STREAM[url] = MediaStreamStatic(url)
                elif isinstance(url, str) and url.lower().startswith("file://"):
                    StreamManager.STREAM[url] = MediaStreamFile(url[7:])
                else:
                    try:
                        StreamManager.STREAM[url] = MediaStreamDevice(url, fps)
                    except Exception as e:
                        logger.error(e)
                        StreamManager.STREAM[url] = MediaStreamURL(url, fps)

                stream = StreamManager.STREAM[url]

                if endpoint is not None and stream.captured:
                    StreamingServer.endpointAdd(endpoint, stream)
                # logger.info("{} {}", stream, url)
            except Exception as e:
                logger.error(str(e))

        return stream

    def pause(self, url: str) -> None:
        if (stream := StreamManager.STREAM.get(url, None)) is None:
            return
        stream.pause()

# =============================================================================
# === SERVER ===
# =============================================================================

class StreamingHandler(BaseHTTPRequestHandler):
    def __init__(self, outputs, *arg, **kw) -> None:
        self.__outputs = outputs
        super().__init__(*arg, **kw)

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
                    logger.error(str(e))
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

class StreamingServer(metaclass=Singleton):
    OUT = {}

    @classmethod
    def endpointAdd(cls, name: str, stream: MediaStreamDevice) -> None:
        StreamingServer.OUT[name] = {'_': stream, 'b': None}
        logger.info(f"ENDPOINT_ADD ({name})")

    def __init__(self, host: str='', port: int=7227) -> None:
        self.__host = host
        self.__port = port
        self.__address = (self.__host, self.__port)
        self.__thread_server = threading.Thread(target=self.__server, daemon=True)
        self.__thread_server.start()
        self.__thread_capture = threading.Thread(target=self.__capture, daemon=True)
        self.__thread_capture.start()
        logger.info("STARTED")

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

def __getattr__(name: str) -> Any:
    if name == "StreamManager":
        return StreamManager()
    elif name == "StreamingServer":
        return StreamingServer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# =============================================================================
# === TESTING ===
# =============================================================================

def streamReadTest() -> None:
    urls = [
        "http://63.142.183.154:6103/mjpg/video.mjpg",
        "https://images.squarespace-cdn.com/content/v1/600743194a20ea052ee04fbb/a1415a46-a24d-4efb-afe2-9ef896d2da62/CircleUnderBerry2.jpg",
        "0", 1,
        "https://images.squarespace-cdn.com/content/v1/600743194a20ea052ee04fbb/1611094440833-JRYTH6W6ODHF0F66FSTI/CD876BD2-EDF9-4FE0-9E85-4D05EEFE9513.jpeg",
        "file://z:\chiggins.png",
        "http://104.207.27.126:8080/mjpg/video.mjpg",
        "http://185.133.99.214:8011/mjpg/video.mjpg",
        "http://tapioles.eu:85/mjpg/video.mjpg",
        "http://63.142.190.238:6106/mjpg/video.mjpg",
        "http://77.222.181.11:8080/mjpg/video.mjpg",
        "http://195.196.36.242/mjpg/video.mjpg",
        "http://honjin1.miemasu.net/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://clausenrc5.viewnetcam.com:50003/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://tamperehacklab.tunk.org:38001/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://vetter.viewnetcam.com:50000/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard",
        "http://webcam.mchcares.com/mjpg/video.mjpg",
        "http://htadmcam01.larimer.org/mjpg/video.mjpg",
        "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg",
        "https://gbpvcam01.taferresorts.com/mjpg/video.mjpg",
        "http://217.147.30.197:8087/mjpg/video.mjpg",
    ]
    streamIdx = 0

    widthT = 160
    heightT = 120

    empty = np.zeros((heightT, widthT, 3), dtype=np.uint8)
    try:
        StreamManager().capture(urls[streamIdx % len(urls)])
    except Exception as e:
        logger.error(e)
    streamIdx += 1

    while True:
        streams = []
        for x in StreamManager().active:
            _, chunk = x.frame
            if chunk is None:
                chunk = np.zeros((heightT, widthT, 3), dtype=np.uint8)
            else:
                chunk = cv2.resize(chunk, (widthT, heightT))
            streams.append(chunk)

        if len(streams) > 0:
            frame = image_grid(streams, widthT, heightT)
        else:
            frame = empty

        try:
            cv2.imshow("Media", frame)
        except Exception as e:
            logger.error(e)
        val = cv2.waitKey(1) & 0xFF
        if val == ord('c'):
            try:
                StreamManager().capture(urls[streamIdx % len(urls)])
            except Exception as e:
                logger.error(e)
            streamIdx += 1
        elif val == ord('q'):
            break

    cv2.destroyAllWindows()

def streamWriteTest() -> None:
    logger.debug(cv2.getBuildInformation())
    ss = StreamingServer()

    fpath = 'res/stream-video.mp4'
    try:
        device = StreamManager().capture(fpath, endpoint=f'/media')
        device.fps = 30
    except:
        pass

    StreamManager().capture(0, endpoint='/stream/0')
    StreamManager().capture(1, endpoint='/stream/1')

    while 1:
        pass

if __name__ == "__main__":
    streamReadTest()
    # streamWriteTest()
    pass