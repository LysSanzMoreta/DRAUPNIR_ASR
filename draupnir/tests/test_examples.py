import sys,os
import pytest
import logging
import subprocess
from pathlib import Path
logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).parents[2] #two levels up dirname

#TODO: Custom example, build True example
CPU_EXAMPLES = ["Draupnir_example.py -use-cuda False -name simulations_insulin_2 -use-custom False -build False -one-hot False -n 1 -guide delta_map -aa-probs 21",
                "Draupnir_example.py -use-cuda False -name simulations_insulin_2 -use-custom False -build False -one-hot False -n 1 -guide variational -aa-probs 21",
                "Draupnir_example.py -use-cuda False -name simulations_insulin_2 -use-custom False -build False -one-hot False -n 1 -guide delta_map -aa-probs 21 -embedding-dim 60",
                "Draupnir_example.py -use-cuda False -name simulations_src_sh3_2 -use-custom False -build False -one-hot False -n 1 -guide delta_map -aa-probs 21 -plate True -plate-size 50 -use-scheduler True",
                ]
GPU_EXAMPLES = ["Draupnir_example.py -use-cuda True -name simulations_src_sh3_1 -use-custom False -build False -one-hot False -n 1 -guide delta_map -aa-probs 21",
                "Draupnir_example.py -use-cuda True -name simulations_src_sh3_1 -use-custom False -build False -one-hot False -n 1 -guide variational -aa-probs 21",
               ]

@pytest.mark.skip(reason="skip")
@pytest.mark.parametrize("example", CPU_EXAMPLES)
def test_cpu_examples(example):
    logger.info("Running:\npython examples/{}".format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLE_DIR, filename)
    subprocess.check_call([sys.executable, filename] + args)

@pytest.mark.parametrize("example", GPU_EXAMPLES)
def test_gpu_examples(example):
    logger.info("Running:\npython examples/{}".format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLE_DIR, filename)
    subprocess.check_call([sys.executable, filename] + args)

if __name__ == '__main__':
    pytest.main(["-s","test_examples.py"]) #-s is to run the statements after yield?