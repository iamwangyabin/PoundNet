import io
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from random import choice, random, randint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def sample_continuous(values):
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return random() * (values[1] - values[0]) + values[0]
    raise ValueError("Expected a list with one or two values.")


def sample_discrete(values):
    if len(values) == 1:
        return values[0]
    return choice(values)


def gaussian_blur(image, sigma):
    return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def cv2_jpg(image, compress_val):
    image_bgr = image[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    _, encoded_image = cv2.imencode(".jpg", image_bgr, encode_param)
    decoded_image = cv2.imdecode(encoded_image, 1)
    return decoded_image[:, :, ::-1]


def pil_jpg(image, compress_val):
    output = BytesIO()
    Image.fromarray(image).save(output, format="jpeg", quality=compress_val)
    output.seek(0)
    return np.array(Image.open(output).copy())


def jpeg_from_key(image, compress_val, key):
    methods = {"cv2": cv2_jpg, "pil": pil_jpg}
    return methods[key](image, compress_val)


class DataAugment:
    def __init__(self, blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual):
        self.blur_prob = blur_prob
        self.blur_sig = blur_sig
        self.jpg_prob = jpg_prob
        self.jpg_method = jpg_method
        self.jpg_qual = jpg_qual

    def __call__(self, image):
        image = np.array(image)
        if random() < self.blur_prob:
            image = gaussian_blur(image, sample_continuous(self.blur_sig))

        if random() < self.jpg_prob:
            image = jpeg_from_key(
                image,
                sample_discrete(self.jpg_qual),
                sample_discrete(self.jpg_method),
            )

        return Image.fromarray(image)


class RandomCompress:
    def __init__(self, method="JPEG", qf=(60, 100)):
        self.method = method
        self.qf = qf

    def __call__(self, image):
        output = io.BytesIO()
        quality = randint(int(self.qf[0]), int(self.qf[1]))
        image.save(output, self.method, quality=quality, optimize=True)
        output.seek(0)
        return Image.open(output).copy()


class Compress:
    def __init__(self, method="JPEG", qf=90):
        self.qf = qf
        self.method = method

    def __call__(self, image):
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, self.method, quality=self.qf, optimize=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream).copy()
