"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Support
"""

import io
import math
import base64
import sys
import urllib
import requests
from enum import Enum
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import cv2
import torch
import cupy as cp
import numpy as np
from numba import jit, cuda
from daltonlens import simulate
from scipy import ndimage
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageOps, ImageChops
from blendmodes.blend import BlendType, blendLayers

from loguru import logger

from Jovimetrix.sup.util import grid_make

