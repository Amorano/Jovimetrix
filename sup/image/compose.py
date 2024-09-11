"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Image Composition Operation Support
"""

from typing import List, Tuple

import cv2
import numpy as np

from loguru import logger

from Jovimetrix.sup.image import MIN_IMAGE_SIZE, TYPE_IMAGE, EnumEdge, \
    EnumInterpolation, TYPE_fCOORD2D, image_crop_center

from Jovimetrix.sup.image.misc import image_detect, image_edge_wrap, image_grayscale

# =============================================================================
# === IMAGE ===
# =============================================================================

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

def image_mask_binary(image: TYPE_IMAGE) -> TYPE_IMAGE:
    """
    Convert an image to a binary mask where non-black pixels are 1 and black pixels are 0.
    Supports BGR, single-channel grayscale, and RGBA images.

    Args:
        image (TYPE_IMAGE): Input image in BGR, grayscale, or RGBA format.

    Returns:
        TYPE_IMAGE: Binary mask with the same width and height as the input image, where
                    pixels are 1 for non-black and 0 for black.
    """
    if image.ndim == 2:
        # Grayscale image
        gray = image
    elif image.shape[2] == 3:
        # BGR image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        # RGBA image
        alpha_channel = image[..., 3]
        # Create a mask from the alpha channel where alpha > 0
        alpha_mask = alpha_channel > 0
        # Convert RGB to grayscale
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Apply the alpha mask to the grayscale image
        gray = cv2.bitwise_and(gray, gray, mask=alpha_mask.astype(np.uint8))
    else:
        raise ValueError("Unsupported image format")

    # Create a binary mask where any non-black pixel is set to 1
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)
    return mask.astype(np.uint8)

def image_minmax(image:List[TYPE_IMAGE]) -> Tuple[int, int, int, int]:
    h_min = w_min = 100000000000
    h_max = w_max = MIN_IMAGE_SIZE
    for img in image:
        if img is None:
            continue
        h, w = img.shape[:2]
        h_max = max(h, h_max)
        w_max = max(w, w_max)
        h_min = min(h, h_min)
        w_min = min(w, w_min)

    # x,y - x+width, y+height
    return w_min, h_min, w_max, h_max

def image_recenter(image: TYPE_IMAGE) -> TYPE_IMAGE:
    cropped_image = image_detect(image)[0]
    new_image = np.zeros(image.shape, dtype=np.uint8)
    paste_x = (new_image.shape[1] - cropped_image.shape[1]) // 2
    paste_y = (new_image.shape[0] - cropped_image.shape[0]) // 2
    new_image[paste_y:paste_y+cropped_image.shape[0], paste_x:paste_x+cropped_image.shape[1]] = cropped_image
    return new_image

def image_rotate(image: TYPE_IMAGE, angle: float, center:TYPE_fCOORD2D=(0.5, 0.5), edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    h, w = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)

    height, width = image.shape[:2]
    c = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(c, -angle, 1.0)
    image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
    if edge != EnumEdge.CLIP:
        image = image_crop_center(image, w, h)
    return image

def image_scale(image: TYPE_IMAGE, scale:TYPE_fCOORD2D=(1.0, 1.0), sample:EnumInterpolation=EnumInterpolation.LANCZOS4, edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:

    h, w = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)

    height, width = image.shape[:2]
    width = int(width * scale[0])
    height = int(height * scale[1])
    image = cv2.resize(image, (width, height), interpolation=sample.value)

    if edge != EnumEdge.CLIP:
        image = image_crop_center(image, w, h)
    return image

def image_translate(image: TYPE_IMAGE, offset: TYPE_fCOORD2D=(0.0, 0.0), edge: EnumEdge=EnumEdge.CLIP, border_value:int=0) -> TYPE_IMAGE:
    """
    Translates an image by a given offset. Supports various edge handling methods.

    Args:
        image (TYPE_IMAGE): Input image as a numpy array.
        offset (TYPE_fCOORD2D): Tuple (offset_x, offset_y) representing the translation offset.
        edge (EnumEdge): Enum representing edge handling method. Options are 'CLIP', 'WRAP', 'WRAPX', 'WRAPY'.

    Returns:
        TYPE_IMAGE: Translated image.
    """

    def translate(img: TYPE_IMAGE) -> TYPE_IMAGE:
        height, width = img.shape[:2]
        scalarX = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 1.0
        scalarY = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 1.0

        M = np.float32([[1, 0, offset[0] * width * scalarX], [0, 1, offset[1] * height * scalarY]])
        if edge == EnumEdge.CLIP:
            border_mode = cv2.BORDER_CONSTANT
        else:
            border_mode = cv2.BORDER_WRAP

        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)

    return translate(image)

def image_transform(image: TYPE_IMAGE, offset:TYPE_fCOORD2D=(0.0, 0.0), angle:float=0, scale:TYPE_fCOORD2D=(1.0, 1.0), sample:EnumInterpolation=EnumInterpolation.LANCZOS4, edge:EnumEdge=EnumEdge.CLIP) -> TYPE_IMAGE:
    sX, sY = scale
    if sX < 0:
        image = cv2.flip(image, 1)
        sX = -sX
    if sY < 0:
        image = cv2.flip(image, 0)
        sY = -sY
    if sX != 1. or sY != 1.:
        image = image_scale(image, (sX, sY), sample, edge)
    if angle != 0:
        image = image_rotate(image, angle, edge=edge)
    if offset[0] != 0. or offset[1] != 0.:
        image = image_translate(image, offset, edge)
    return image
