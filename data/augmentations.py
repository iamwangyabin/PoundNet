import io
from io import BytesIO
import cv2
import numbers
import numpy as np
from collections.abc import Sequence
from PIL import ImageFile, Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Compress:
    def __init__(self, method="JPEG", qf=90):
        self.qf = qf
        self.method = method

    def __call__(self, image):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, self.method, quality=self.qf, optimize=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)


