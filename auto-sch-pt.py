# auto-scheduler code pytorch framework
 
import os # adding this line
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata
from tvm.contrib import utils, graph_executor as runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import os # adding this line
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/" 
import onnx
import readline
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
from tvm.contrib import utils, graph_executor as runtime
import time
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
target = tvm.target.Target("cuda")

dev = tvm.device(str(target), 0)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

######################################################################
# Create standard optimize engine

tmp = utils.tempdir()
lib_fname = tmp.relpath("/mnt/36548d81-5509-40eb-ab7b-4616b3bd3f03/TVM/tvm/engine1-autosch_gpu_unopt_resnet50.tar")
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


############################################################################################
log_file = "network1.json"
# Extract tasks from the network
print("Extract tasks...")
#mod, params = relay.frontend.from_onnx(onnx_model, shape_dict) 

#mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

run_tuning()


######################################################################
# .. note:: Explain the printed information during tuning
#
#   During the tuning, a lot of information will be printed on the console.
#   They are used for debugging purposes. The most important info is the output
#   of the task scheduler. The following table is a sample output.
#
#   .. code-block:: c
#
#     ----------------------------------------------------------------------
#     ------------------------------  [ Task Scheduler ]
#     ----------------------------------------------------------------------
#     |  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
#     -------------------------------------------------
#     |    0 |        0.005 |           0.88 |     64 |
#     |    1 |        0.010 |          99.10 |     64 |
#     |    2 |        0.006 |           0.00 |     64 |
#     |    3 |        0.145 |         979.78 |    384 |
#     |    4 |        0.130 |        1097.02 |    384 |
#     |    5 |        0.143 |         992.69 |    384 |
#     |    6 |        0.076 |        1526.86 |    192 |
#     |    7 |        0.115 |         999.44 |    320 |
#     |    8 |        0.079 |        1449.39 |    320 |
#     |    9 |        0.122 |         938.73 |    384 |
#     |   10 |        0.063 |        1832.98 |    192 |
#     |   11 |        0.072 |        1763.62 |    256 |
#     |   12 |        0.062 |        2036.40 |    192 |
#     |   13 |        0.068 |        1874.44 |    192 |
#     |   14 |        0.049 |        2346.50 |    128 |
#     |   15 |        0.076 |        1694.31 |    256 |
#     |   16 |        0.067 |        1933.30 |    448 |
#     |   17 |        0.076 |        1680.90 |    256 |
#     |   18 |        0.022 |          98.43 |     64 |
#     |   19 |        0.076 |        3112.55 |    192 |
#     |   20 |        0.013 |        2026.44 |     64 |
#     |   21 |        0.011 |        1136.69 |     64 |
#     |   22 |        0.013 |         992.47 |     64 |
#     |   23 |        0.020 |         627.56 |     64 |
#     -------------------------------------------------
#     Estimated total latency: 1.587 ms  Trials: 4992  Used time : 13296 s  Next ID: 3
#
#   This table lists the latency and (estimated) speed of all tasks.
#   It also lists the allocation of measurement trials for all tasks.
#   The last line prints the total weighted latency of these tasks,
#   which can be a rough estimation of the end-to-end execution time
#   of the network.
#   The last line also prints the total number of measurement trials,
#   total time spent on auto-tuning and the id of the next task to tune.
#
#   There will also be some "tvm::Error"s and CUDA errors, because the
#   auto-scheduler will try some invalid schedules.
#   You can safely ignore them if the tuning can continue, because these
#   errors are isolated from the main process.
#

######################################################################
# .. note:: Terminate the tuning earlier
#
#   You can terminate the tuning earlier by forcibly killing this process.
#   As long as you get at least one valid schedule for each task in the log file,
#   you should be able to do the compilation (the secion below).
#


#################################################################
# Compile and Evaluate
# --------------------
# After auto-tuning, we can compile the network with the best schedules we found.
# All measurement records are dumped into the log file during auto-tuning,
# so we can read the log file and load the best schedules.

# Compile with the history best
tvm_start=time.time()
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input(input_name, data_tvm)
tvm_end= time.time()
time_diff= tvm_end-tvm_start
print("The time to build optimized model is",time_diff)


exe = relay.vm.compile(mod, target, params=params)
#vm = profiler_vm.VirtualMachineProfiler(exe, dev)
# report = vm.profile([data], func_name="main", number=100, repeat=3, end_to_end=True)
#report = vm.profile(data=data)
#print(report)

###############################################################################################################33\\
#tvm_start=time.time()
#input_name = "data"
#shape_dict = {input_name: img_data.shape}

#mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
#target = tvm.target.Target("cuda")
#with tvm.transform.PassContext(opt_level=3):
#    lib = relay.build(mod, target=target, params=params)
#tvm_end= time.time()
#time_diff= tvm_end-tvm_start
#print("The time to build optimized model is",time_diff)

tmp = utils.tempdir()
lib_fname = tmp.relpath("/mnt/36548d81-5509-40eb-ab7b-4616b3bd3f03/TVM/tvm/engine1_autosch_gpu_opt_resnet50.tar")
lib.export_library(lib_fname)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=500, min_repeat_ms=500))

print("unoptimized: %s" % (unoptimized))
