import torch

def to_device(i, d):
    if isinstance(i, torch.Tensor):
        return i.to(device=d)
    elif isinstance(i, (tuple, list)):
        return tuple(to_device(e, d) for e in i)
    else:
        raise RuntimeError('inputs are weird')

class GPUWrapper(torch.nn.Module):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.device = f'cuda:{torch.version.interp % torch.cuda.device_count()}'
        # self.stream = torch.cuda.Stream(self.device)
        # with torch.cuda.stream(self.stream):
        self.root.to(device=self.device)

    def __getstate__(self):
        return self.root

    __setstate__ = __init__

    def forward(self, *args):
        # with torch.cuda.stream(self.stream):
        iput = to_device(args, self.device)
        return to_device(self.root(*iput), 'cpu')


if __name__ == '__main__':
    def check_close(a, b):
        if isinstance(a, (list, tuple)):
            for ae, be in zip(a, b):
                check_close(ae, be)
        else:
            print(torch.max(torch.abs(a - b)))
            assert torch.allclose(a, b)

    import sys
    from torch.package import PackageImporter
    i = PackageImporter(sys.argv[1])
    torch.version.interp = 0
    model = i.load_pickle('model', 'model.pkl')
    eg = i.load_pickle('model', 'example.pkl')
    r = model(*eg)

    gpu_model = GPUWrapper(model)
    r2 = gpu_model(*eg)
    check_close(r, r2)
