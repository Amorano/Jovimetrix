"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Media Stream Support
"""

import os
import sys
import json
import time
import threading
from typing import Any
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mss
import mss.tools
import numpy as np
from PIL import Image, ImageGrab
from loguru import logger

from Jovimetrix import Singleton, MIN_IMAGE_SIZE
from Jovimetrix.sup.image import image_grid, image_load, pil2cv

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
# === SCREEN / WINDOW CAPTURE ===
# =============================================================================

def monitor_capture_all(width:int=None, height:int=None) -> cv2.Mat:
    img = ImageGrab.grab(all_screens=True)
    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if height is not None and width is not None:
        return cv2.resize(img, (width, height))
    return img

def monitor_capture(monitor:int=0, tlwh:tuple[int, int, int, int]=None, width:int=None, height:int=None) -> cv2.Mat:
    with mss.mss() as sct:
        if tlwh is not None:
            region = {'top': tlwh[0], 'left': tlwh[1], 'width': tlwh[2], 'height': tlwh[3]}
            img = sct.grab(region)
        else:
            monitor = sct.monitors[monitor]
            img = sct.grab(monitor)

        img = np.array(img, dtype=np.uint8)
        if height is not None and width is not None:
            img = cv2.resize(img, (width, height))
        return img

def monitor_list() -> dict:
    ret = {}
    with mss.mss() as sct:
        ret = {i:v for i, v in enumerate(sct.monitors)}
    return ret

if sys.platform.startswith('win'):

    import win32gui
    import win32ui
    from ctypes import windll

    def window_list() -> dict:
        _windows = {}
        def window_enum_handler(hwnd, ctx) -> None:
            if win32gui.IsWindowVisible(hwnd):
                name = win32gui.GetWindowText(hwnd).strip()
                if name != "" and name not in ['Calculator', 'Program Manager', 'Settings', 'Microsoft Text Input Application']:
                    _windows[hwnd] = name

        win32gui.EnumWindows(window_enum_handler, None)
        return _windows

    def window_capture(hwnd:int, dpi:bool=True, clientOnly:bool=True) -> cv2.Mat:
        # hwnd = win32gui.FindWindow(None, 'Calculator')
        if dpi:
            windll.user32.SetProcessDPIAware()

        try:
            if clientOnly:
                left, top, right, bot = win32gui.GetClientRect(hwnd)
            else:
                left, top, right, bot = win32gui.GetWindowRect(hwnd)
        except:
            return

        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)
        im = pil2cv(im)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        try:
            saveDC.DeleteDC()
        except:
            pass

        try:
            mfcDC.DeleteDC()
        except:
            pass

        win32gui.ReleaseDC(hwnd, hwndDC)
        return im

# =============================================================================
# === MEDIA ===
# =============================================================================

def camera_list() -> list:
    camera_list = {}
    failed = 0
    idx = 0
    while failed < 2:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            camera_list[idx] = {
                'w': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'h': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS))
            }
            cap.release()
        else:
            failed += 1
        idx += 1
    return camera_list

class MediaStreamBase:

    TIMEOUT = 5.

    def __init__(self, fps:float=30) -> None:
        self.__quit = False
        self.__paused = False
        self.__captured = False
        self.__ret = False
        self.__fps = fps
        self.__timeout = None
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
                        self.__quit = True
                        break

                    self.__paused = pause
                    self.__captured = True
                    logger.info(f"CAPTURED")

                if self.__timeout is None:
                    self.__timeout = time.perf_counter() + self.TIMEOUT

                # call the run capture frame command on subclasses
                self.__ret, newframe = self.callback()
                if newframe is not None:
                    self.__frame = newframe
                    self.__timeout = None

            if self.__timeout is not None and time.perf_counter() > self.__timeout:
                self.__timeout = None
                self.__quit = True
                logger.warning(f"TIMEOUT")

            waste = max(waste - time.perf_counter(), 0)
            time.sleep(waste)

        logger.info(f"STOPPED")
        self.end()

    def __del__(self) -> None:
        self.end()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def callback(self) -> tuple[bool, Any]:
        return True, None

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
    def frame(self) -> tuple[bool, Any]:
        return self.__ret, self.__frame

    @property
    def fps(self) -> float:
        return self.__fps

    @fps.setter
    def fps(self, val: float) -> None:
        self.__fps = max(1, val)

class MediaStreamStatic(MediaStreamBase):
    """A stream coming from ComfyUI."""
    def __init__(self) -> None:
        self.image = None
        super().__init__()

    def callback(self) -> tuple[bool, Any]:
        return True, self.image

class MediaStreamDesktop(MediaStreamBase):
    """Desktop, specific monitor, specific window or cropped region."""
    def __init__(self, area:int|str=None, region:tuple[int, int, int, int]=None) -> None:
        try: area = int(area)
        except: pass
        super().__init__()

    def callback(self) -> tuple[bool, Any]:
        return True, None

class MediaStreamURL(MediaStreamBase):
    """A media point (could be a camera index)."""
    def __init__(self, url:int|str, fps:float=30) -> None:
        self.__url = url
        try: self.__url = int(url)
        except: pass
        self.__source = None
        self.__last = None
        super().__init__(fps)

    def callback(self) -> tuple[bool, Any]:
        ret = False
        try:
            ret, result = self.__source.read()
        except:
            pass

        if ret:
            self.__last = result
            return ret, result

        count = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        pos = int(self.source.get(cv2.CAP_PROP_POS_FRAMES))
        if pos >= count:
            self.source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, result = self.__source.read()

        # maybe its a single frame -- if we ever got one.
        if not ret and self.__last is not None:
            return True, self.__last

        return ret, result

    @property
    def url(self) -> str:
        return self.__url

    @property
    def source(self) -> cv2.VideoCapture:
        return self.__source

    def capture(self) -> bool:
        if self.captured:
            return True
        self.__source = cv2.VideoCapture(self.__url, cv2.CAP_ANY)
        if self.captured:
            time.sleep(0.3)
            return True
        return False

    @property
    def captured(self) -> bool:
        if self.__source is None:
            return False
        return self.__source.isOpened()

    def release(self) -> None:
        if self.__source is not None:
            self.__source.release()
        super().release()

class MediaStreamDevice(MediaStreamURL):
    """A system device like a web camera."""
    def __init__(self, url:int|str, fps:float=30) -> None:
        self.__focus = 0
        self.__exposure = 1
        self.__zoom = 0
        super().__init__(url, fps=fps)

    @property
    def zoom(self) -> float:
        return self.__zoom

    @zoom.setter
    def zoom(self, val: float) -> None:
        self.__zoom = np.clip(val, 0, 1)
        val = 100 + 300 * self.__zoom
        self.source.set(cv2.CAP_PROP_ZOOM, val)

    @property
    def exposure(self) -> float:
        return self.__exposure

    @exposure.setter
    def exposure(self, val: float) -> None:
        # -10 to -1 range
        self.__exposure = np.clip(val, 0, 1)
        val = -10 + 9 * self.__exposure
        self.source.set(cv2.CAP_PROP_EXPOSURE, val)

    @property
    def focus(self) -> float:
        return self.__focus

    @focus.setter
    def focus(self, val: float) -> None:
        self.__focus = np.clip(val, 0, 1)
        val = 255 * self.__focus
        self.source.set(cv2.CAP_PROP_FOCUS, val)

class MediaStreamFile(MediaStreamBase):
    """A file served from a local file using file:// as the 'uri'."""
    def __init__(self, url:str) -> None:
        self.__image = image_load(url)[0]
        super().__init__()

    def callback(self) -> tuple[bool, Any]:
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
                    StreamManager.STREAM[url] = MediaStreamStatic()
                elif isinstance(url, str) and url.lower().startswith("file://"):
                    StreamManager.STREAM[url] = MediaStreamFile(url[7:])

                else:
                    try:
                        url = int(url)
                        StreamManager.STREAM[url] = MediaStreamDevice(url, fps=fps)
                    except Exception as _:
                        StreamManager.STREAM[url] = MediaStreamURL(url, fps=fps)

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
                time.sleep(0.001)

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
            time.sleep(0.001)

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
        "https://images.squarespace-cdn.com/content/v1/600743194a20ea052ee04fbb/a1415a46-a24d-4efb-afe2-9ef896d2da62/CircleUnderBerry2.jpg",
        "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
        "https://images.squarespace-cdn.com/content/v1/600743194a20ea052ee04fbb/a1415a46-a24d-4efb-afe2-9ef896d2da62/CircleUnderBerry2.jpg",
        "file://z:\\alex.png",
        "0", 1,
        "https://images.squarespace-cdn.com/content/v1/600743194a20ea052ee04fbb/1611094440833-JRYTH6W6ODHF0F66FSTI/CD876BD2-EDF9-4FE0-9E85-4D05EEFE9513.jpeg",
        "http://63.142.183.154:6103/mjpg/video.mjpg",
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

def capture_screen_test() -> None:
    while(True):
        img = window_capture(monitor=33, width=960, height=540)
        cv2.imshow('window', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    streamReadTest()
    # streamWriteTest()
    for m, v in camera_list().items():
        print(m, v)
    exit()
    capture_screen_test()
    pass