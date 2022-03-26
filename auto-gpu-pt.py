
import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata
from tvm.contrib import utils, graph_executor as runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

# PyTorch imports
import torch
import torchvision

######################################################################
# Load a pretrained PyTorch model
# -------------------------------
model_name = "resnet50"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

######################################################################
# Load a test image
# -----------------
# Classic cat example!
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = tvm.target.cuda()

dev = tvm.device(str(target), 0)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

######################################################################
# Create standard optimize engine

tmp = utils.tempdir()
lib_fname = tmp.relpath("/mnt/36548d81-5509-40eb-ab7b-4616b3bd3f03/TVM/tvm/tmp/eng1_gpu_unopt_resnet50.tar")
lib.export_library(lib_fname)


######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
# Execute

import time

since = time.time()
for i in range(500):
    first_pass=time.time()
    m.run()
    last_pass=time.time() - first_pass
    print("",last_pass)
time_elapsed = time.time() - since
print('Time elapsed for unoptimized inference is',time_elapsed)

# Get outputs
tvm_output = m.get_output(0)

################################################################################
# Collect Basic Performance Data
# ------------------------------
# We want to collect some basic performance data associated with this
# unoptimized model and compare it to a tuned model later. To help account for
# CPU noise, we run the computation in multiple batches in multiple
# repetitions, then gather some basis statistics on the mean, median, and
# standard deviation.
import timeit


since=time.time()
timing_number = 10
timing_repeat = 10
unoptimized = (
   np.array(timeit.Timer(lambda: m.run()).repeat(repeat=timing_repeat, number=timing_number))
   * 1000
   / timing_number
)

print("The UNOPTIMIZED ONE IS",unoptimized)
unoptimized = {
   "mean": np.mean(unoptimized),
   "median": np.median(unoptimized),
   "std": np.std(unoptimized),
}

now=time.time()
time_elapsed= now-since
print("The time for basic performance data is",time_elapsed)
print(unoptimized)
####################################################################
# Postprocess the output
# ---------------------------------

class_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
class_path = download_testdata(class_url, "synset.txt", module="data")

with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

### Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.numpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

print("top-1 id: {}, class name: {}".format(top1_tvm, tvm_class_key))


################################################################################
# Tune the model
# --------------
import tvm.auto_scheduler as auto_scheduler
#from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm


# # Create a simple structure for holding tuning options. We use an XGBoost
# # algorithim for guiding the search. For a production job, you will want to set
# # the number of trials to be larger than the value of 10 used here. For CPU we
# # recommend 1500, for GPU 3000-4000. The number of trials required can depend
# # on the particular model and processor, so it's worth spending some time
# # evaluating performance across a range of values to find the best balance
# # between tuning time and model optimization. Because running tuning is time
# # intensive we set number of trials to 10, but do not recommend a value this
# # small. The ``early_stopping`` parameter is the minimum number of trails to
# # run before a condition that stops the search early can be applied. The
# # measure option indicates where trial code will be built, and where it will be
# # run. In this case, we're using the ``LocalRunner`` we just created and a
# # ``LocalBuilder``. The ``tuning_records`` option specifies a file to write
# # the tuning data to.

autostart_time=time.time()

tuning_option = {
    "tuner": "ga",
    "trials": 100,
    "early_stopping": 10,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), 
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
    "tuning_records": "squ.json",
}

# ################################################################################
# # .. admonition:: Defining the Tuning Search Algorithm
# #
# #   By default this search is guided using an `XGBoost Grid` algorithm.
# #   Depending on your model complexity and amount of time available, you might
# #   want to choose a different algorithm.


# ################################################################################
# # .. admonition:: Setting Tuning Parameters
# #
# #   In this example, in the interest of time, we set the number of trials and
# #   early stopping to 10. You will likely see more performance improvements if
# #   you set these values to be higher but this comes at the expense of time
# #   spent tuning. The number of trials required for convergence will vary
# #   depending on the specifics of the model and the target platform.

# # begin by extracting the tasks from the onnx model

print("Extract tasks...")
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

print("above for loop")

# # Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    #tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj = GATuner(task, pop_size=50)
    #tuner_obj = RandomTuner(task)
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )


# ################################################################################
# # The output from this tuning process will look something like this:
# #
# # .. code-block:: bash
# #
# #   # [Task  1/24]  Current/Best:   10.71/  21.08 GFLOPS | Progress: (60/1000) | 111.77 s Done.
# #   # [Task  1/24]  Current/Best:    9.32/  24.18 GFLOPS | Progress: (192/1000) | 365.02 s Done.
# #   # [Task  2/24]  Current/Best:   22.39/ 177.59 GFLOPS | Progress: (960/1000) | 976.17 s Done.
# #   # [Task  3/24]  Current/Best:   32.03/ 153.34 GFLOPS | Progress: (800/1000) | 776.84 s Done.
# #   # [Task  4/24]  Current/Best:   11.96/ 156.49 GFLOPS | Progress: (960/1000) | 632.26 s Done.
# #   # [Task  5/24]  Current/Best:   23.75/ 130.78 GFLOPS | Progress: (800/1000) | 739.29 s Done.
# #   # [Task  6/24]  Current/Best:   38.29/ 198.31 GFLOPS | Progress: (1000/1000) | 624.51 s Done.
# #   # [Task  7/24]  Current/Best:    4.31/ 210.78 GFLOPS | Progress: (1000/1000) | 701.03 s Done.
# #   # [Task  8/24]  Current/Best:   50.25/ 185.35 GFLOPS | Progress: (972/1000) | 538.55 s Done.
# #   # [Task  9/24]  Current/Best:   50.19/ 194.42 GFLOPS | Progress: (1000/1000) | 487.30 s Done.
# #   # [Task 10/24]  Current/Best:   12.90/ 172.60 GFLOPS | Progress: (972/1000) | 607.32 s Done.
# #   # [Task 11/24]  Current/Best:   62.71/ 203.46 GFLOPS | Progress: (1000/1000) | 581.92 s Done.
# #   # [Task 12/24]  Current/Best:   36.79/ 224.71 GFLOPS | Progress: (1000/1000) | 675.13 s Done.
# #   # [Task 13/24]  Current/Best:    7.76/ 219.72 GFLOPS | Progress: (1000/1000) | 519.06 s Done.
# #   # [Task 14/24]  Current/Best:   12.26/ 202.42 GFLOPS | Progress: (1000/1000) | 514.30 s Done.
# #   # [Task 15/24]  Current/Best:   31.59/ 197.61 GFLOPS | Progress: (1000/1000) | 558.54 s Done.
# #   # [Task 16/24]  Current/Best:   31.63/ 206.08 GFLOPS | Progress: (1000/1000) | 708.36 s Done.
# #   # [Task 17/24]  Current/Best:   41.18/ 204.45 GFLOPS | Progress: (1000/1000) | 736.08 s Done.
# #   # [Task 18/24]  Current/Best:   15.85/ 222.38 GFLOPS | Progress: (980/1000) | 516.73 s Done.
# #   # [Task 19/24]  Current/Best:   15.78/ 203.41 GFLOPS | Progress: (1000/1000) | 587.13 s Done.
# #   # [Task 20/24]  Current/Best:   30.47/ 205.92 GFLOPS | Progress: (980/1000) | 471.00 s Done.
# #   # [Task 21/24]  Current/Best:   46.91/ 227.99 GFLOPS | Progress: (308/1000) | 219.18 s Done.
# #   # [Task 22/24]  Current/Best:   13.33/ 207.66 GFLOPS | Progress: (1000/1000) | 761.74 s Done.
# #   # [Task 23/24]  Current/Best:   53.29/ 192.98 GFLOPS | Progress: (1000/1000) | 799.90 s Done.
# #   # [Task 24/24]  Current/Best:   25.03/ 146.14 GFLOPS | Progress: (1000/1000) | 1112.55 s Done.

# ################################################################################
# # Compiling an Optimized Model with Tuning Data
# # ----------------------------------------------
# #
# # As an output of the tuning process above, we obtained the tuning records
# # stored in ``resnet-50-v2-autotuning.json``. The compiler will use the results to
# # generate high performance code for the model on your specified target.
# #
# # Now that tuning data for the model has been collected, we can re-compile the
# # model using optimized operators to speed up our computations.

with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3, config={}):
        lib = relay.build(mod, target=target, params=params)

autotvm_end_time=time.time()

time_diff=autotvm_end_time-autostart_time
print("The time to create the optimized model is", time_diff)


tmp = utils.tempdir()
lib_fname = tmp.relpath("/mnt/36548d81-5509-40eb-ab7b-4616b3bd3f03/TVM/tvm/tmp/eng1_gpu_opt_resnet50.tar")
lib.export_library(lib_fname)

dev = tvm.device(str(target), 0)

from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
m.set_input(input_name, tvm.nd.array(img.astype(dtype)))

# Execute
since = time.time()
for i in range(500):
    first_pass=time.time()
    m.run()
    last_pass=time.time() - first_pass
    print("",last_pass)
time_elapsed = time.time() - since
print('Time elapsed for optimized inference is',time_elapsed)

# Get outputs
tvm_output = m.get_output(0)


####################################################################
# Postprocess the output
# ---------------------------------

class_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
class_path = download_testdata(class_url, "synset.txt", module="data")

with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

### Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.numpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

print("top-1 id: {}, class name: {}".format(top1_tvm, tvm_class_key))


# ################################################################################
# # Comparing the Tuned and Untuned Models

import timeit

timing_number = 10
timing_repeat = 10
optimized = (
   np.array(timeit.Timer(lambda: m.run()).repeat(repeat=timing_repeat, number=timing_number))
   * 1000
   / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}

print("optimized: %s" % (optimized))
print("unoptimized: %s" % (unoptimized))