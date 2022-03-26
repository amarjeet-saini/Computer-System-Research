import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

model_path = 'squeezenet1.1-7.onnx'
session = ort.InferenceSession(model_path)

images=[]
names=[]

for filename in os.listdir('data_clean'):
    img = cv2.imread(os.path.join('data_clean',filename))
    if img is not None:
        images.append(img)
        names.append(filename)

print("loaded {} images".format(len(images)))

rows = open('synset.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:] for r in rows]

data=[]
inf_time = []

for i in range(0,len(images)):
    with Image.open("data_clean/"+names[i]) as img:
        img = np.array(img.convert('RGB'))
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)    

    ort_inputs = {session.get_inputs()[0].name: img}
    tic = timer()
    preds = session.run(None, ort_inputs)[0]
    toc = timer()
    
    inf_time.append(toc - tic)
    
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    data.append((names[i], preds[a[0]],classes[a[0]]))

total = 0

for x in inf_time:    
    total += x

print(f"inference time : {total}")

# write result to file
import sys
sys.stdout = open('r_clean.txt','wt')
for i in range(0,len(data)):
    print(data[i])





