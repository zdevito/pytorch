"""
Generate the example files that torchpy_test uses.
"""
from pathlib import Path
import torch
import torchvision

from torch.package import PackageExporter

from examples import Simple, rand, Sequence, SNLIClassifier, SuperRes, Config

from torchvision.models.squeezenet import squeezenet1_1


def save(name, model, model_jit, eg):
    with PackageExporter(str(p / name)) as e:
        e.save_pickle('model', 'model.pkl', model)
        e.save_pickle('model', 'example.pkl', eg)
    model_jit.save(str(p / (name + '_jit')))

if __name__ == "__main__":
    p = Path(__file__).parent / "generated"
    p.mkdir(exist_ok=True)

    resnet = torchvision.models.resnet18()
    resnet.eval()
    resnet_eg = torch.rand(1, 3, 224, 224)
    resnet_traced = torch.jit.trace(resnet, resnet_eg)
    save('resnet', resnet, resnet_traced, (resnet_eg,))

    simple = Simple(10, 20)
    save('simple', simple, torch.jit.script(simple), (torch.rand(10, 20),))

    save('rand', rand, torch.jit.script(rand), ())

    sequence = Sequence()
    save('time_sequence', sequence, torch.jit.script(sequence), (torch.rand(3, 4),))

    sn = squeezenet1_1()
    save('squeezenet_model', sn, torch.jit.script(sn), (resnet_eg,))

    # snli
    premise = torch.LongTensor(48, 64).random_(0, 100)
    hypothesis = torch.LongTensor(24, 64).random_(0, 100)
    snli = SNLIClassifier(Config())
    save('snli', snli, torch.jit.script(snli), (premise, hypothesis))

    # superresolution
    sr = SuperRes(upscale_factor=4)
    sr_input = torch.rand(5, 1, 32, 32)
    save('super_resolution', sr, torch.jit.script(sr), (sr_input,))
