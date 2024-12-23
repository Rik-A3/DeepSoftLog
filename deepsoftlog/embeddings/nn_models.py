import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel


class AdditionFunctor(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        result = _add_probs(x[0], x[1])
        for t in x[2:]:
            result = _add_probs(result, t)
        return result


def _add_probs(x1, x2):
    result = torch.zeros(10).to(x1.device)
    for i in range(10):
        result += x1[i] * torch.roll(x2, i, 0)
    return result


class CarryFunctor(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        c1 = _carry_probs(x[0], x[1])
        if len(x) == 2:
            return c1
        a1 = _add_probs(x[0], x[1])
        c2 = _carry_probs(a1, x[2])
        result = torch.zeros(10).to(c1.device)
        result[0] = c1[0] * c2[0]
        result[1] = 1 - result[0]
        return result


def _carry_probs(x1, x2):
    result = torch.zeros(10).to(x1.device)
    result[0] = (torch.cumsum(x2, 0).flip((0,)) * x1).sum()
    result[1] = 1 - result[0]
    return result

class EmbeddingFunctor(nn.Module):
    def __init__(self, arity=1, ndims=128):
        super().__init__()
        hidden_dims = max(128, ndims)
        self.model = nn.Sequential(
            nn.Linear(arity * ndims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, ndims),
        )
        self.activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.model(x.flatten())
        return self.activation(x)


class LeNet5(nn.Module):
    """
    LeNet5. A small convolutional network.
    """

    def __init__(self, output_features=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 1 28 28 -> 6 24 24
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_features),
        )
        self.activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)[0]
        return self.activation(x)
        #return

class Llama31_8B(nn.Module):
    """
    TODO
    """

    def __init__(self):
        pass # TODO

    def forward(self, x):
        pass # TODO

    def eval(self):
        pass # TODO

MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

def hash_tensor(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    while x.ndim > 0:
        x = _reduce_last_axis(x)
    return x.item()

@torch.no_grad()
def _reduce_last_axis(x: Tensor) -> Tensor:
    assert x.dtype == torch.int64
    acc = torch.zeros_like(x[..., 0])
    for i in range(x.shape[-1]):
        acc *= MULTIPLIER
        acc += INCREMENT
        acc += x[..., i]
        # acc %= MODULUS  # Not really necessary.
    return acc

class XLMRobertaLarge(nn.Module):
    """
    https://huggingface.co/FacebookAI/xlm-roberta-large
    """

    def __init__(self, ndims=1024):
        super().__init__()

        self._tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = AutoModel.from_pretrained("xlm-roberta-large")
        for name,param in self.model.encoder.named_parameters():
            if int(name.split(".")[1]) < 22: # All but the last layer
                param.requires_grad = False
        self.embedding_cache = {}
        self.cache_count = 10

    def half_precision(self):
        self.model.half()

        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    def forward(self, x):
        if hash_tensor(x) not in self.embedding_cache:
            self.embedding_cache[hash_tensor(x)] = self._forward(x)

        return self.embedding_cache[hash_tensor(x)]

    def _forward(self, x):
        tokens = x[:, 0, :]
        attention_mask = x[:, 1, :]

        embedding = self.model(tokens, attention_mask)
        pooler_output = embedding.pooler_output

        return torch.softmax(pooler_output, dim=1)

    def reset_cache(self):
        self.embedding_cache = {}


if __name__ == "__main__":
    model = XLMRobertaLarge()
    embedding = model(["Hello, my dog is cute.", "Hello, my cat is cute."])
    print(embedding)
