from tvm.driver import tvmc
import os

dataDir = os.getcwd() + "/data/"
model_filename   = dataDir   + 'resnet50-v1-7.onnx'
records_filename = dataDir   + 'tune_resnet50_1_3_224_224.json'


def tune_model():
    model = tvmc.load(
        model_filename,
        shape_dict={'data' : [1, 3, 224, 224]}
    )

    # Simple skylake CPU
    tvmc.tune(
        model,
        target = "llvm -mcpu=skylake",
        enable_autoscheduler = True,
        tuning_records=records_filename
    )

tune_model()
