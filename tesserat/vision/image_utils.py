import cv2
import numpy as np
from pdf2image import convert_from_path

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    return denoised

def pdf_to_images(pdf_path: str):
    return convert_from_path(pdf_path)
