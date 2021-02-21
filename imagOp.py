# image processing

import numpy as np
from PIL import Image, ImageStat, ImageOps, ImageDraw, ImageFont

def generateImageSeq(imgs, stamps=None, stampsColor=(255, 255, 255), stampsSize=64, split=0, scaleBar=None, orient=1):
    # to generate Img sequence, with optional stamps on the top
    # split specifies the can be something in the middle
    # orietation 0 is horizontal, 1 is vertical
    # scaleBar (10, 100, '100um') - 10um/px, 100um, marked as "100um"
    # example: generateImageSeq(items, stamps=stamps, split=10)
    imgSize = list(np.shape(Image.open(imgs[0])))
    finImgSize = imgSize.copy()
    # finImgSize[orient] = finImgSize[orient]*(len(imgs)) + split*(len(imgs)-1)
    finImgSize[orient] = 0
    splitSize = imgSize.copy()
    if split > 0:
        splitSize[orient] = split
        splitImg = np.ones(splitSize)*255
    else:
        splitImg = None
    finImg = np.empty(finImgSize)
    fnt = ImageFont.truetype('Arial.ttf', stampsSize)
    
#     "Pillow/Tests/fonts/FreeMono.ttf",
    for i, imagei in enumerate(imgs):
        pic = Image.open(imagei)
        if stamps is not None:
            ImageDraw.Draw(pic).text(
                (imgSize[0]/20, imgSize[1]/40),  # Coordinates
                stamps[i],  # Text
                fill=stampsColor,  # Color
                font=fnt,
            )
        if scaleBar is not None:
            ImageDraw.Draw(pic).line(
                (imgSize[0]*5/6, imgSize[1]*23/40, imgSize[0]*5/6+scaleBar[1]/scaleBar[0], imgSize[1]*23/40),
                fill=stampsColor,
                width=5,
            )
            ImageDraw.Draw(pic).text(
                (imgSize[0]*4/5, imgSize[1]*25/40),  # Coordinates
                scaleBar[2],  # Text
                fill=stampsColor,  # Color
                font=fnt,
            )
        if (splitImg is not None) and (i > 0):
            finImg = np.append(finImg, splitImg, axis=orient)
        finImg = np.append(finImg, np.array(pic), axis=orient)
    return Image.fromarray(np.uint8(finImg))
