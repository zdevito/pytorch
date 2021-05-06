import sys
print("I AM ALIVE!")
print(id(None))
# sys.path.append('/private/home/zdevito/miniconda3/envs/py38/lib/python3.8/site-packages/')
# sys.path.append('/raid/zdevito/vision')
sys.path.extend(['', '/private/home/zdevito/miniconda3/envs/py38/lib/python38.zip', '/private/home/zdevito/miniconda3/envs/py38/lib/python3.8', '/private/home/zdevito/miniconda3/envs/py38/lib/python3.8/lib-dynload', '/private/home/zdevito/miniconda3/envs/py38/lib/python3.8/site-packages', '/raid/zdevito/pytorch', '/private/home/zdevito/miniconda3/envs/py38/lib/python3.8/site-packages/dataclasses-0.8-py3.8.egg', '/raid/zdevito/vision', '/private/home/zdevito/miniconda3/envs/py38/lib/python3.8/site-packages/Pillow-8.0.1-py3.8-linux-x86_64.egg', '/raid/zdevito/benchmark/torchbenchmark/models/maskrcnn_benchmark'])
import regex
print("regex imported, running tests...")
from unittest import main
# main(exit=False, module='test_regex')
import torch
import torchvision

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda()
model.eval()
x = [torch.rand(3, 300, 400).cuda(), torch.rand(3, 500, 400).cuda()]
predictions = model(x)
print(predictions)
# import numpy as np

# a = np.arange(15).reshape(3, 5)
# print(a + a)