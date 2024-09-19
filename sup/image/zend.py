
import math
import urllib
from typing import List, Tuple, Any

import cv2
import torch
import requests
import numpy as np
from numba import jit
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageChops, ImageOps

from loguru import logger

from Jovimetrix.sup.image import TYPE_IMAGE, TYPE_PIXEL, TYPE_iRGB, \
    cv2pil, image_convert, image_matte, pil2cv

from Jovimetrix.sup.image.channel import channel_add

from Jovimetrix.sup.image.color import image_grayscale

from Jovimetrix.sup.util import grid_make

def image_crop_head(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """
    Given a file path or np.ndarray image with a face,
    returns cropped np.ndarray around the largest detected
    face.

    Parameters
    ----------
    - `path_or_array` : {`str`, `np.ndarray`}
        * The filepath or numpy array of the image.

    Returns
    -------
    - `image` : {`np.ndarray`, `None`}
        * A cropped numpy array if face detected, else None.
    """

    MIN_FACE = 8

    gray = image_grayscale(image)
    h, w = image.shape[:2]
    minface = int(np.sqrt(h**2 + w**2) / MIN_FACE)

    '''
        # Create the haar cascade
        face_cascade = cv2.CascadeClassifier(self.casc_path)

        # ====== Detect faces in the image ======
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(minface, minface),
            flags=cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,
        )

        # Handle no faces
        if len(faces) == 0:
            return None

        # Make padding from biggest face found
        x, y, w, h = faces[-1]
        pos = self._crop_positions(
            img_height,
            img_width,
            x,
            y,
            w,
            h,
        )

        # ====== Actual cropping ======
        image = image[pos[0] : pos[1], pos[2] : pos[3]]

        # Resize
        if self.resize:
            with Image.fromarray(image) as img:
                image = np.array(img.resize((self.width, self.height)))

        # Underexposition fix
        if self.gamma:
            image = check_underexposed(image, gray)
        return bgr_to_rbg(image)

    def _determine_safe_zoom(self, imgh, imgw, x, y, w, h):
        """
        Determines the safest zoom level with which to add margins
        around the detected face. Tries to honor `self.face_percent`
        when possible.

        Parameters:
        -----------
        imgh: int
            Height (px) of the image to be cropped
        imgw: int
            Width (px) of the image to be cropped
        x: int
            Leftmost coordinates of the detected face
        y: int
            Bottom-most coordinates of the detected face
        w: int
            Width of the detected face
        h: int
            Height of the detected face

        Diagram:
        --------
        i / j := zoom / 100

                  +
        h1        |         h2
        +---------|---------+
        |      MAR|GIN      |
        |         (x+w, y+h)|
        |   +-----|-----+   |
        |   |   FA|CE   |   |
        |   |     |     |   |
        |   ├──i──┤     |   |
        |   |  cen|ter  |   |
        |   |     |     |   |
        |   +-----|-----+   |
        |   (x, y)|         |
        |         |         |
        +---------|---------+
        ├────j────┤
                  +
        """
        # Find out what zoom factor to use given self.aspect_ratio
        corners = itertools.product((x, x + w), (y, y + h))
        center = np.array([x + int(w / 2), y + int(h / 2)])
        i = np.array(
            [(0, 0), (0, imgh), (imgw, imgh), (imgw, 0), (0, 0)]
        )  # image_corners
        image_sides = [(i[n], i[n + 1]) for n in range(4)]

        corner_ratios = [self.face_percent]  # Hopefully we use this one
        for c in corners:
            corner_vector = np.array([center, c])
            a = distance(*corner_vector)
            intersects = list(intersect(corner_vector, side) for side in image_sides)
            for pt in intersects:
                if (pt >= 0).all() and (pt <= i[2]).all():  # if intersect within image
                    dist_to_pt = distance(center, pt)
                    corner_ratios.append(100 * a / dist_to_pt)
        return max(corner_ratios)

    def _crop_positions(
        self,
        imgh,
        imgw,
        x,
        y,
        w,
        h,
    ):
        """
        Retuns the coordinates of the crop position centered
        around the detected face with extra margins. Tries to
        honor `self.face_percent` if possible, else uses the
        largest margins that comply with required aspect ratio
        given by `self.height` and `self.width`.

        Parameters:
        -----------
        imgh: int
            Height (px) of the image to be cropped
        imgw: int
            Width (px) of the image to be cropped
        x: int
            Leftmost coordinates of the detected face
        y: int
            Bottom-most coordinates of the detected face
        w: int
            Width of the detected face
        h: int
            Height of the detected face
        """
        zoom = self._determine_safe_zoom(imgh, imgw, x, y, w, h)

        # Adjust output height based on percent
        if self.height >= self.width:
            height_crop = h * 100.0 / zoom
            width_crop = self.aspect_ratio * float(height_crop)
        else:
            width_crop = w * 100.0 / zoom
            height_crop = float(width_crop) / self.aspect_ratio

        # Calculate padding by centering face
        xpad = (width_crop - w) / 2
        ypad = (height_crop - h) / 2

        # Calc. positions of crop
        h1 = x - xpad
        h2 = x + w + xpad
        v1 = y - ypad
        v2 = y + h + ypad

        return [int(v1), int(v2), int(h1), int(h2)]
    '''

def image_detect(image: TYPE_IMAGE) -> Tuple[TYPE_IMAGE, Tuple[int, ...]]:
    gray = image_grayscale(image)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the item we want to recenter
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, (x, y, w, h)

def image_diff(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, threshold:int=0,
               color:TYPE_PIXEL=(255, 0, 0)) -> Tuple[TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, TYPE_IMAGE, float]:
    """imageA, imageB, diff, thresh, score
    """
    h1, w1 = imageA.shape[:2]
    h2, w2 = imageB.shape[:2]
    w1 = max(w1, w2)
    h1 = max(h1, h2)
    imageA = image_matte(imageA, (0, 0, 0, 0), w1, h1)
    imageA = image_convert(imageA, 3)
    imageB = image_matte(imageB, (0, 0, 0, 0), w1, h1)
    imageB = image_convert(imageB, 3)
    grayA = image_grayscale(imageA)
    grayB = image_grayscale(imageB)
    (score, diff) = ssim(grayA, grayB, full=True, channel_axis=2)
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    high_a = imageA.copy()
    high_a = image_convert(high_a, 3)
    high_b = imageB.copy()
    high_b = image_convert(high_b, 3)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(imageA, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(imageB, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(high_a, [c], 0, color[::-1], -1)
        cv2.drawContours(high_b, [c], 0, color[::-1], -1)
        cv2.drawContours(diff_box, [c], 0, color[::-1], -1)
    imageA = cv2.addWeighted(imageA, 0.0, high_a, 1, 0)
    imageB = cv2.addWeighted(imageB, 0.0, high_b, 1, 0)
    return imageA, imageB, diff, thresh, score

def image_disparity(imageA: np.ndarray) -> np.ndarray:
    imageA = imageA.astype(np.float32) / 255.
    imageA = cv2.normalize(imageA, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    disparity_map = np.divide(1.0, imageA, where=imageA != 0)
    return np.where(imageA == 0, 1, disparity_map)

def image_histogram_statistics(histogram:np.ndarray, L=256)-> TYPE_IMAGE:
    sumPixels = np.sum(histogram)
    normalizedHistogram = histogram/sumPixels
    mean = 0
    for i in range(L):
        mean += i * normalizedHistogram[i]
    variance = 0
    for i in range(L):
        variance += (i-mean)**2 * normalizedHistogram[i]
    std = np.sqrt(variance)
    return mean, variance, std

def image_gradient_map2(image, gradient_map):
    na = np.array(image)
    grey = np.mean(na, axis=2).astype(np.uint8)
    cmap = np.array(gradient_map.convert('RGB'))
    result = np.zeros((*grey.shape, 3), dtype=np.uint8)
    grey_reshaped = grey.reshape(-1)
    np.take(cmap.reshape(-1, 3), grey_reshaped, axis=0, out=result.reshape(-1, 3))
    return result

def image_grid(data: List[TYPE_IMAGE], width: int, height: int) -> TYPE_IMAGE:
    #@TODO: makes poor assumption all images are the same dimensions.
    chunks, col, row = grid_make(data)
    frame = np.zeros((height * row, width * col, 4), dtype=np.uint8)
    i = 0
    for y, strip in enumerate(chunks):
        for x, item in enumerate(strip):
            cc = item.shape[2] if item.ndim == 3 else 1
            if cc == 3:
                item = channel_add(item)
            y1, y2 = y * height, (y+1) * height
            x1, x2 = x * width, (x+1) * width
            frame[y1:y2, x1:x2, ] = item
            i += 1

    return frame

def image_merge(imageA: TYPE_IMAGE, imageB: TYPE_IMAGE, axis: int=0,
                flip: bool=False) -> TYPE_IMAGE:
    if flip:
        imageA, imageB = imageB, imageA
    axis = 1 if axis == "HORIZONTAL" else 0
    return np.concatenate((imageA, imageB), axis=axis)

def image_recenter(image: TYPE_IMAGE) -> TYPE_IMAGE:
    cropped_image = image_detect(image)[0]
    new_image = np.zeros(image.shape, dtype=np.uint8)
    paste_x = (new_image.shape[1] - cropped_image.shape[1]) // 2
    paste_y = (new_image.shape[0] - cropped_image.shape[0]) // 2
    new_image[paste_y:paste_y+cropped_image.shape[0], paste_x:paste_x+cropped_image.shape[1]] = cropped_image
    return new_image

def image_stereo_shift(image: TYPE_IMAGE, depth: TYPE_IMAGE, shift:float=10) -> TYPE_IMAGE:
    # Ensure base image has alpha
    image = image_convert(image, 4)
    depth = image_convert(depth, 1)
    deltas = np.array((depth / 255.0) * float(shift), dtype=int)
    shifted_data = np.zeros(image.shape, dtype=np.uint8)
    _, width = image.shape[:2]
    for y, row in enumerate(deltas):
        for x, dx in enumerate(row):
            x2 = x + dx
            if (x2 >= width) or (x2 < 0):
                continue
            shifted_data[y][x2] = image[y][x]

    shifted_image = cv2pil(shifted_data)
    alphas_image = Image.fromarray(
        ndimage.binary_fill_holes(
            ImageChops.invert(
                shifted_image.getchannel("A")
            )
        )
    ).convert("1")
    shifted_image.putalpha(ImageChops.invert(alphas_image))
    return pil2cv(shifted_image)

# KERNELS

def MEDIAN3x3(image: TYPE_IMAGE) -> TYPE_IMAGE:
    height, width = image.shape[:2]
    out = np.zeros([height, width])
    for i in range(1, height-1):
        for j in range(1, width-1):
            temp = [
                image[i-1, j-1],
                image[i-1, j],
                image[i-1, j + 1],
                image[i, j-1],
                image[i, j],
                image[i, j + 1],
                image[i + 1, j-1],
                image[i + 1, j],
                image[i + 1, j + 1]
            ]

            temp = sorted(temp)
            out[i, j]= temp[4]
    return out

def kernel(stride: int) -> TYPE_IMAGE:
    """
    Generate a kernel matrix with a specific stride.

    The kernel matrix has a size of (stride, stride) and is filled with values
    such that if i < j, the element is set to -1; if i > j, the element is set to 1.

    Parameters:
    - stride (int): The size of the square kernel matrix.

    Returns:
    - TYPE_IMAGE: The generated kernel matrix.

    Example:
    >>> KERNEL(3)
    array([[ 0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0]], dtype=int8)
    """
    # Create an initial matrix of zeros
    kernel = np.zeros((stride, stride), dtype=np.int8)

    # Create a mask for elements where i < j and set them to -1
    mask_lower = np.tril(np.ones((stride, stride), dtype=bool), k=-1)
    kernel[mask_lower] = -1

    # Create a mask for elements where i > j and set them to 1
    mask_upper = np.triu(np.ones((stride, stride), dtype=bool), k=1)
    kernel[mask_upper] = 1

    return kernel

#
#
#

def image_load_exr(url: str) -> Tuple[TYPE_IMAGE, TYPE_IMAGE]:
    """
    exr_file     = OpenEXR.InputFile(url)
    exr_header   = exr_file.header()
    r,g,b = exr_file.channels("RGB", pixel_type=Imath.PixelType(Imath.PixelType.FLOAT) )

    dw = exr_header[ "dataWindow" ]
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1

    image = np.ones( (h, w, 4), dtype = np.float32 )
    image[:, :, 0] = np.core.multiarray.frombuffer( r, dtype = np.float32 ).reshape(h, w)
    image[:, :, 1] = np.core.multiarray.frombuffer( g, dtype = np.float32 ).reshape(h, w)
    image[:, :, 2] = np.core.multiarray.frombuffer( b, dtype = np.float32 ).reshape(h, w)
    return create_optix_image_2D( w, h, image.flatten() )
    """
    pass

def image_load_from_url(url: str, stream:bool=True) -> TYPE_IMAGE:
    """Creates a CV2 BGR image from a url."""
    try:
        image  = urllib.request.urlopen(url)
        image = np.asarray(bytearray(image.read()), dtype=np.uint8)
        return cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        try:
            image = Image.open(requests.get(url, stream=stream).raw)
            return pil2cv(image)
        except Exception as e:
            logger.error(str(e))

def image_save_gif(fpath:str, images: List[Image.Image], fps: int=0,
                loop:int=0, optimize:bool=False) -> None:

    fps = min(50, max(1, fps))
    images[0].save(
        fpath,
        append_images=images[1:],
        duration=3, # int(100.0 / fps),
        loop=loop,
        optimize=optimize,
        save_all=True
    )

def image_load_data(data: str) -> TYPE_IMAGE:
    img = ImageOps.exif_transpose(data)
    return pil2cv(img)

def image_gradient(width:int, height:int, color_map:dict=None) -> TYPE_IMAGE:
    if color_map is None:
        color_map = {0: (0,0,0,255)}
    else:
        color_map = {np.clip(float(k), 0, 1): [np.clip(int(c), 0, 255) for c in v] for k, v in color_map.items()}
    color_map = dict(sorted(color_map.items()))
    image = Image.new('RGBA', (width, height))
    draw = image.load()
    widthf = float(width)

    @jit
    def gaussian(x, a, b, c, d=0) -> Any:
        return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

    def pixel(x, spread:int=1) -> TYPE_iRGB:
        ws = widthf / (spread * len(color_map))
        r = sum([gaussian(x, p[0], k * widthf, ws) for k, p in color_map.items()])
        g = sum([gaussian(x, p[1], k * widthf, ws) for k, p in color_map.items()])
        b = sum([gaussian(x, p[2], k * widthf, ws) for k, p in color_map.items()])
        return min(255, int(r)), min(255, int(g)), min(255, int(b))

    for x in range(width):
        r, g, b = pixel(x)
        for y in range(height):
            draw[x, y] = r, g, b
    return pil2cv(image)

def torch_rgb2hsv(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    if (cc := rgb.shape[2]) == 3:
        hsv_h = torch.empty_like(rgb[:, 0:1, :])
    elif cc == 4:
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

def torch_hsv2rgb(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

def torch_rgb2hsl(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.
    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)

def torch_hsl2rgb(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb
