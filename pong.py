# from PIL import ImageGrab
# import cv2
# import pytesseract
# import numpy as np
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
# ask = input("Do you want to ocr in realtime or choose a picture (r/p)?")
# if ask == 'r':
#     while True:
#         screen = np.array(ImageGrab.grab(bbox=(700, 300, 1600, 1000)))
#         # print('Frame took {} seconds'.format(time.time()-last_time))
#         cv2.imshow('window', screen)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break
#         print(pytesseract.image_to_string(screen, lang='eng', config='--psm 6'))
# else:
#     Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
#     filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
#     print(pytesseract.image_to_string(filename, lang='eng', config='--psm 6'))

import imutils
import numpy as np
from PIL import ImageGrab, Image
import cv2
import time
import argparse
import os
import pytesseract
from ctypes import windll
from matplotlib import pyplot as plt

user32 = windll.user32
user32.SetProcessDPIAware()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def process_img(screen):
    global filename, ctr
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
                    help="type of preprocessing to be done")
    args = vars(ap.parse_args())
    b = screen.copy()

    resizedimg = cv2.resize(b, (1000, 666))

    gamma = .3  # change the value here to get different result
    adjusted = adjust_gamma(resizedimg, gamma=gamma)
    ret, thresh = cv2.threshold(adjusted, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(thresh, kernel, iterations=1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args["preprocess"] == "thresh":
        img_gray = cv2.threshold(img_gray, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    elif args["preprocess"] == "blur":
        img_gray = cv2.medianBlur(img_gray, 3)

    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    gaussian_3 = cv2.GaussianBlur(blurred, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(blurred, 1.5, gaussian_3, -0.5, 0, blurred)

    resizedimg1 = cv2.resize(unsharp_image, (int(1000 / 3), int(666 / 3)))
    gaussian_3 = cv2.GaussianBlur(resizedimg1, (5, 5), 0)

    resizedimg2 = cv2.resize(gaussian_3, (1000, 666))

    unsharp_image1 = cv2.addWeighted(unsharp_image, 1.5, resizedimg2, -0.5, 0, unsharp_image)
    img = imutils.rotate(unsharp_image1, 6.5)
    gamma = .3  # change the value here to get different result
    adjusted = adjust_gamma(img, gamma=gamma)
    ret, thresh = cv2.threshold(adjusted, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(thresh, kernel, iterations=1)

    im2, ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    filename = os.getpid()
    fug = 0
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = img[y:y + h, x:x + w]

        edged = cv2.Canny(roi, 100, 200)
        # show ROI
        # cv2.imwrite('roi_imgs.png', roi)
        cv2.imshow('charachter' + str(i), edged)
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)
        cool = str(filename) + str(fug) + '.png'
        cv2.imwrite(cool, edged)
        fug += 1
    img = cv2.Canny(img, 100, 200)

    cv2.imshow('marked areas', img)
    return fug


prevtext = int(100)
while True:

    global filename, ctr
    screen = np.array(ImageGrab.grab(bbox=(790, 1315, 839, 1345)))
    # print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    new_screen = process_img(screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    time.sleep(.5)
    text = ''
    for g in range(new_screen):
        cool = str(filename) + str(g) + '.png'
        pravtext = pytesseract.image_to_string(Image.open(cool), lang='eng', config='--psm 10 --oem 3 -c '
                                                                                    'tessedit_char_whitelist=0123456789')
        os.remove(cool)
        text += pravtext
    print(text)


