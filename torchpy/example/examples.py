import torch
import torch.nn as nn

class Simple(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        output = self.weight + input
        return output

def rand():
    return torch.rand(3, 4)

class Sequence(torch.nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input):
        # TODO: add future as input with default val
        # see https://github.com/pytorch/pytorch/issues/8724
        outputs = torch.empty((3, 0))
        h_t = torch.zeros((3, 51))
        c_t = torch.zeros((3, 51))
        h_t2 = torch.zeros((3, 51))
        c_t2 = torch.zeros((3, 51))

        output = torch.zeros([3, 51])
        future = 2

        # TODO: chunk call should appear as the for loop iterable
        # We hard-code it to 4 for now.
        a, b, c, d = input.chunk(input.size(1), dim=1)
        for input_t in (a, b, c, d):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs = torch.cat((outputs, output), 1)
        for _ in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs = torch.cat((outputs, output), 1)
        return outputs


class Linear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, input):
        if len(input.size()) <= 2:
            return self.linear.forward(input)
        size = input.size()[:2]
        out = self.linear.forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.n_cells = self.config.n_cells
        self.d_hidden = self.config.d_hidden
        self.birnn = self.config.birnn
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=dropout,
                           bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.n_cells, batch_size, self.d_hidden
        h0 = c0 = inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

class SNLIClassifier(nn.Module):
    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2 * config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size] * 2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, config.d_out))

    def forward(self, premise, hypothesis):
        prem_embed = self.embed(premise)
        hypo_embed = self.embed(hypothesis)
        prem_embed = prem_embed.detach()
        hypo_embed = hypo_embed.detach()
        prem_embed = self.relu(self.projection(prem_embed))
        hypo_embed = self.relu(self.projection(hypo_embed))
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores

class Config:
    n_embed = 100
    d_embed = 100
    d_proj = 300
    dp_ratio = 0.0  # For deterministic testing TODO: change by fixing seed in checkTrace?
    d_hidden = 30
    birnn = True
    d_out = 300
    fix_emb = True
    projection = True
    n_layers = 2
    n_cells = 4  # 2 * n_layers because birnn = True

class SuperRes(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x
