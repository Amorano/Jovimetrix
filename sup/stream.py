"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Media Stream Support
"""

import os
import ssl
import sys
import json
import time
import array
import threading
from typing import Any, List, Tuple
from itertools import repeat
from configparser import ConfigParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mss
import mss.tools
import numpy as np
from PIL import Image, ImageGrab

from comfy.cli_args import args as cmd_args

from loguru import logger

# SPOUT SUPPORT
JOV_SPOUT = os.getenv("JOV_SPOUT", 'true').strip().lower() in ('true', '1', 't')
if JOV_SPOUT:
    try:
        import SpoutGL
        from OpenGL import GL
        logger.info("SPOUT GL SUPPORT")
    except Exception as e:
        logger.error("NO SPOUT GL SUPPORT")
        logger.error(e)
else:
    logger.warning("SKIPPING SPOUT GL SUPPORT")

from .. import Singleton

from .image import TYPE_PIXEL, \
    image_load, pil2cv

# ==============================================================================

class StreamMissingException(Exception): pass

# ==============================================================================
# === GLOBAL CONFIG ===
# ==============================================================================

cfg = ConfigParser()
JOV_SCAN_DEVICES = True
JOV_SCAN_DEVICES = os.getenv("JOV_SCAN_DEVICES", "True").lower() in ['1', 'true', 'on']
JOV_STREAM_HOST = os.getenv("JOV_STREAM_HOST", '')
JOV_STREAM_PORT = 7227
try:
    JOV_STREAM_PORT = int(os.getenv("JOV_STREAM_PORT", JOV_STREAM_PORT))
except Exception as e:
    logger.error(str(e))

# ==============================================================================
# === SCREEN / WINDOW CAPTURE ===
# ==============================================================================

def monitor_capture_all(width:int=None, height:int=None) -> cv2.Mat:
    img = ImageGrab.grab(all_screens=True)
    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if height is not None and width is not None:
        return cv2.resize(img, (width, height))
    return img

def monitor_capture(monitor:int=0, tlwh:Tuple[int, int, int, int]=None, width:int=None, height:int=None) -> cv2.Mat:
    with mss.mss() as sct:
        region = sct.monitors[monitor]
        if tlwh is not None:
            t, l, w, h = region['top'], region['left'], region['width'], region['height']
            l += min(tlwh[1], tlwh[2]) * w
            t += min(tlwh[0], tlwh[3]) * h
            w = abs(tlwh[2] - tlwh[1]) * w
            h = abs(tlwh[3] - tlwh[0]) * h
            region = {'top': int(t), 'left': int(l), 'width': int(w), 'height': int(h)}

        img = sct.grab(region)
        img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if height is not None and width is not None:
            img = cv2.resize(img, (width, height))
        return img

def monitor_list() -> dict:
    ret = {}
    with mss.mss() as sct:
        ret = {i:v for i, v in enumerate(sct.monitors)}
    return ret

def window_list() -> dict:
    return {}

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

    def window_capture(hwnd:str, dpi:bool=True, clientOnly:bool=True) -> cv2.Mat:
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
            bmpstr, 'raw', 'RGBX', 0, 1)
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

elif sys.platform.startswith('darwin'):
    from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll
    import Quartz

    def get_window_dimensions(hwnd):
        window_info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionIncludingWindow, hwnd)

        for window_info in window_info_list:
            window_id = window_info[Quartz.kCGWindowNumber]
            if window_id == hwnd:
                bounds = window_info[Quartz.kCGWindowBounds]
                width = bounds['Width']
                height = bounds['Height']
                left = bounds['X']
                top = bounds['Y']
                return {"top": top, "left": left, "width": width, "height": height}

        return None

    def window_list() -> dict:
        _windows = {}
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        for window in window_list:
            hwnd = window[Quartz.kCGWindowNumber]
            name = window.get(Quartz.kCGWindowName, 'Unnamed Window')
            if name and name not in ['Dock', 'Menu Bar']:
                _windows[hwnd] = name
        return _windows

    def window_capture(hwnd:str, dpi:bool=True, clientOnly:bool=True) -> np.ndarray:
        dimensions = get_window_dimensions(hwnd)
        if dimensions is None:
            return None

        """
        rect = CGRectMake(dimensions['left'], dimensions['top'], dimensions['width'], dimensions['height'])

        # Capture the window image
        image_ref = CGWindowListCreateImage(rect, Quartz.kCGWindowListOptionIncludingWindow, hwnd, Quartz.kCGWindowImageOptionNone)

        if image_ref is None:
            return None

        # Convert Quartz image to a numpy array (you may need to install pyobjc-framework-Quartz)
        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)
        color_space = Quartz.CGColorSpaceCreateDeviceRGB()

        # Create a bitmap context for pixel data
        context = Quartz.CGBitmapContextCreate(None, width, height, 8, width * 4, color_space, Quartz.kCGImageAlphaPremultipliedFirst)
        Quartz.CGContextDrawImage(context, Quartz.CGRectMake(0, 0, width, height), image_ref)

        # Get the pixel data from the context
        data = Quartz.CGBitmapContextGetData(context)
        image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

        # Convert from BGRA to BGR
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        """

        with mss() as sct:
            screenshot = np.array(sct.grab(dimensions))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

# ==============================================================================
# === MEDIA ===
# ==============================================================================

def camera_list() -> list:
    camera_list = {}
    global JOV_SCAN_DEVICES

    if not JOV_SCAN_DEVICES:
        return camera_list
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

                if self.__timeout is None and self.TIMEOUT > 0:
                    self.__timeout = time.perf_counter() + self.TIMEOUT

                # call the run capture frame command on subclasses
                newframe = self.callback()
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

    def callback(self) -> Tuple[bool, Any]:
        return None

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
    def frame(self) -> Any:
        return self.__frame

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

    def callback(self) -> Tuple[bool, Any]:
        return self.image

class MediaStreamURL(MediaStreamBase):
    """A media point (could be a camera index)."""
    def __init__(self, url:int|str, fps:float=30) -> None:
        self.__url = url
        try: self.__url = int(url)
        except: pass
        self.__source = None
        self.__last = None
        super().__init__(fps)

    def callback(self) -> Tuple[bool, Any]:
        ret = False
        try:
            ret, result = self.__source.read()
        except:
            pass

        if ret:
            self.__last = result
            return result

        count = int(self.source.get(cv2.CAP_PROP_FRAME_COUNT))
        pos = int(self.source.get(cv2.CAP_PROP_POS_FRAMES))
        if pos >= count:
            self.source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, result = self.__source.read()

        # maybe its a single frame -- if we ever got one.
        if not ret and self.__last is not None:
            return self.__last

        return result

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
        if self.source is None:
            return
        self.__zoom = np.clip(val, 0, 1)
        val = 100 + 300 * self.__zoom
        self.source.set(cv2.CAP_PROP_ZOOM, val)

    @property
    def exposure(self) -> float:
        return self.__exposure

    @exposure.setter
    def exposure(self, val: float) -> None:
        if self.source is None:
            return
        # -10 to -1 range
        self.__exposure = np.clip(val, 0, 1)
        val = -10 + 9 * self.__exposure
        self.source.set(cv2.CAP_PROP_EXPOSURE, val)

    @property
    def focus(self) -> float:
        return self.__focus

    @focus.setter
    def focus(self, val: float) -> None:
        if self.source is None:
            return
        self.__focus = np.clip(val, 0, 1)
        val = 255 * self.__focus
        self.source.set(cv2.CAP_PROP_FOCUS, val)

if JOV_SPOUT:
    class MediaStreamSpout(MediaStreamBase):
        """Capture from SpoutGL stream."""

        TIMEOUT = 0

        def __init__(self, url:str, fps:float=30) -> None:
            self.__buffer = None
            self.__url = url
            self.__last = None
            self.__width = self.__height = 0
            self.__spout = SpoutGL.SpoutReceiver()
            self.__spout.setReceiverName(url)
            super().__init__(fps)

        def callback(self) -> Any:
            if self.__spout.isUpdated():
                self.__width = self.__spout.getSenderWidth()
                self.__height = self.__spout.getSenderHeight()
                self.__buffer = array.array('B', repeat(0, self.__width * self.__height * 4))
            result = self.__spout.receiveImage(self.__buffer, GL.GL_RGBA, False, 0)
            if self.__buffer is not None and result: # and not SpoutGL.helpers.isBufferEmpty(self.__buffer):
                self.__last = np.asarray(self.__buffer, dtype=np.uint8).reshape((self.__height, self.__width, 4))
                # self.__last[:, :, [0, 2]] = self.__last[:, :, [2, 0]]
            return self.__last

        @property
        def url(self) -> str:
            return self.__url

        @url.setter
        def url(self, url:str) -> None:
            self.__spout.setReceiverName(url)
            self.__url = url

        def __del__(self) -> None:
            if self.__spout is not None:
                self.__spout.ReleaseReceiver()
            self.__spout = None
            del self.__spout

class MediaStreamFile(MediaStreamBase):
    """A file served from a local file using file:// as the 'uri'."""
    def __init__(self, url:str) -> None:
        self.__image, mask = image_load(url)[0]
        super().__init__()

    def callback(self) -> Tuple[bool, Any]:
        return True, self.__image

class StreamManager(metaclass=Singleton):
    STREAM = {}
    def __del__(self) -> None:
        if StreamManager:
            for c in StreamManager.STREAM.values():
                del c

    @property
    def streams(self) -> List[str|int]:
        return list(StreamManager.STREAM.keys())

    @property
    def active(self) -> List[MediaStreamDevice]:
        return [stream for stream in StreamManager.STREAM.values() if stream.captured]

    def frame(self, url: str) -> Any:
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

# ==============================================================================
# === SERVER ===
# ==============================================================================

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

    def __init__(self, host: str='', port: int=JOV_STREAM_PORT) -> None:
        self.__host = host
        self.__port = port
        self.__address = (self.__host, self.__port)
        self.__tls_keyfile = cmd_args.tls_keyfile
        self.__tls_certfile = cmd_args.tls_certfile
        self.__thread_server = threading.Thread(target=self.__server, daemon=True)
        self.__thread_server.start()
        self.__thread_capture = threading.Thread(target=self.__capture, daemon=True)
        self.__thread_capture.start()
        logger.info("STARTED")

    def __server(self) -> None:
        handler = lambda *args: StreamingHandler(StreamingServer.OUT, *args)
        httpd = ThreadingHTTPServer(self.__address, handler)

        if self.__tls_keyfile and self.__tls_certfile:
            # HTTPS mode
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.__tls_certfile, self.__tls_keyfile)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            logger.info(f"HTTPS server {self.__address}")
        else:
            # HTTP mode
            logger.info(f"HTTP server {self.__address}")

        while True:
            httpd.handle_request()

    def __capture(self) -> None:
        while True:
            current = StreamingServer.OUT.copy()
            for k, v in current.items():
                if (device := v['_']) is not None:
                    StreamingServer.OUT[k]['b'] = device.frame
            time.sleep(0.001)

# ==============================================================================
# === SPOUT SERVER ===
# ==============================================================================

if JOV_SPOUT:
    class SpoutSender:
        def __init__(self, host: str='', fps:int=30, frame:TYPE_PIXEL=None) -> None:
            self.__fps = self.__width = self.__height = 0
            self.__frame = None
            self.frame = frame
            self.__host = host
            self.__delay = 0
            self.fps = max(1, fps)
            self.__sender = SpoutGL.SpoutSender()
            self.__sender.setSenderName(self.__host)
            self.__thread_server = threading.Thread(target=self.__server, daemon=True)
            self.__thread_server.start()
            logger.info("STARTED")

        @property
        def frame(self) -> TYPE_PIXEL:
            return self.__frame

        @frame.setter
        def frame(self, image: TYPE_PIXEL) -> None:
            """Must be RGBA"""
            self.__frame = image

        @property
        def host(self) -> str:
            return self.__host

        @host.setter
        def host(self, host: str) -> None:
            if host != self.__host:
                self.__sender = SpoutGL.SpoutSender()
                self.__sender.setSenderName(host)
                self.__host = host

        @property
        def fps(self) -> int:
            return self.__fps

        @fps.setter
        def fps(self, fps: int) -> None:
            self.__fps = max(1, fps)
            self.__delay = 1. / fps

        def __server(self) -> None:
            while 1:
                if self.__sender is not None:
                    if self.__frame is not None:
                        h, w = self.__frame.shape[:2]
                        self.__sender.sendImage(self.__frame, w, h, GL.GL_RGBA, False, 0)
                    self.__sender.setFrameSync(self.__host)
                time.sleep(self.__delay)

        def __del__(self) -> None:
            self.__sender = None
            del self.__sender

def __getattr__(name: str) -> Any:
    if name == "StreamManager":
        return StreamManager()
    elif name == "StreamingServer":
        return StreamingServer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
