import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
from tqdm import tqdm

for im in tqdm(os.listdir('images/imageSet')):
    img = Image.open('images/imageSet/' + im).convert("RGB")
    width, height, _ = np.array(img).shape

    d = ImageDraw.Draw(img)
    for i in range(random.randint(5, 50)):
        fnt = ImageFont.truetype('font.ttf', size=np.random.randint(20, 40))
        fnt.size = np.random.randint(40, 125)
        if(random.randint(5,50)>25):
            d.line((random.random() * width, random.random() * height, random.random() * width, random.random() * height), fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), width = int(random.random()*5))
        d.text((random.random() * width, random.random() * height), ''.join([random.choice(string.digits + string.ascii_letters) for x in range(20)]), fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), font=fnt)
    img.save('output/' + im)