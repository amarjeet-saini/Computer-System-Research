############################################################################################################
# Program to calculate accuracy and inference time of 5000 images for Keras pre-trained model
# Input: .trt engine, labels.txt, images dir
# Output: .txt file
# Flow : keras -> onnx -> .trt engine
# Author: Amarjeet Saini
# Original source : https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/3.%20Using%20Tensorflow%202%20through%20ONNX.ipynb

import os
import numpy as np
import sys
import progressbar
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32
BATCH_SIZE = 1

images=[]
names=[]

# set input_batch for bytes

img_path = 'data_clean/n03100240-ILSVRC2012_val_00022130.JPEG'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
input_batch = np.array(np.repeat(np.expand_dims(np.array(x, dtype=np.float32), axis=0), BATCH_SIZE, axis=0), dtype=np.float32)
input_batch = input_batch.astype(target_dtype)

# load 5000 images from data_clean

for filename in os.listdir('data_clean'):
    img = cv2.imread(os.path.join('data_clean',filename))
    if img is not None:
        images.append(img)
        names.append(filename)

print("loaded {} images".format(len(images)))

rows = open('synset.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:] for r in rows]

data=[]

# load .trt engine

f = open("keras/resnet50/resnet50-dla1_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()


output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) # Need to set output dtype to FP16 to enable FP16

# Allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

inf_time = []

print("-------- start inference -----------")
start = time.time()
bar = progressbar.ProgressBar(maxval=len(images), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for i in range(0,len(images)):
    
    img_path = "data_clean/"+names[i] 
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    input_batch = np.array(np.repeat(np.expand_dims(np.array(x, dtype=np.float32), axis=0), BATCH_SIZE, axis=0), dtype=np.float32)
    input_batch = input_batch.astype(target_dtype)

    cuda.memcpy_htod_async(d_input, input_batch, stream)
    
    tic = time.time() 
    
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    
    toc = time.time()
    inf_time.append(toc-tic)

    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()

    trt_predictions = output.astype(np.float32)
    indices = (-trt_predictions[0]).argsort()[:5]
    # append infernce output to data list
    data.append((names[i], output[0][indices[0]],classes[indices[0]]))

    bar.update(i+1)

bar.finish()
end = time.time()
print("inferernce took {:.5} seconds".format(end - start))

# inference time
total = 0

for x in inf_time:
    total += x 

print(f"inferernce took w/o memcopy and image load {total}")

# write result to file

import sys
sys.stdout = open('resnet50_clean.txt','wt')
for i in range(0,len(data)):
    print(data[i])
