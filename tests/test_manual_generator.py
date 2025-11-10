import torch
import pytest

from src.manual_generation import ManualGenerator


class DummyTokenizer:
    """Minimal tokenizer to support ManualGenerator tests."""

    def __init__(self):
        self.eos_token_id = 0
        self.bos_token_id = 1
        self._vocab = {'<eos>': self.eos_token_id, '<bos>': self.bos_token_id}
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

    def _get_token_id(self, ch: str) -> int:
        if ch not in self._vocab:
            token_id = len(self._vocab)
            self._vocab[ch] = token_id
            self._inv_vocab[token_id] = ch
        return self._vocab[ch]

    def __call__(self, text: str, return_tensors: str = "pt"):
        token_ids = [self._get_token_id(ch) for ch in text]
        if not token_ids:
            token_ids = [self.eos_token_id]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            chars.append(self._inv_vocab.get(token_id, '?'))
        return ''.join(chars)


class DummyOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class DummyModel(torch.nn.Module):
    """Records inputs and returns deterministic logits."""

    def __init__(self, vocab_size: int = 32, fixed_token: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.fixed_token = fixed_token
        self.calls = []

    @property
    def device(self):
        return torch.device('cpu')

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        call_info = {
            "input_ids": input_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone() if attention_mask is not None else None,
            "position_ids": position_ids.detach().clone() if position_ids is not None else None,
            "past_length": past_length,
        }
        self.calls.append(call_info)

        batch_size, seq_len = input_ids.shape
        total_len = past_length + seq_len

        logits = torch.full((batch_size, seq_len, self.vocab_size), -1e9)
        logits[:, :, self.fixed_token] = 0.0

        if use_cache:
            k = torch.zeros(batch_size, 1, total_len, 1)
            v = torch.zeros(batch_size, 1, total_len, 1)
            past = ((k, v),)
        else:
            past = None

        return DummyOutput(logits=logits, past_key_values=past)

    @staticmethod
    def make_past(cache_length: int) -> tuple:
        k = torch.zeros(1, 1, cache_length, 1)
        v = torch.zeros(1, 1, cache_length, 1)
        return ((k, v),)


@pytest.fixture
def generator_with_cache():
    tokenizer = DummyTokenizer()
    model = DummyModel()
    generator = ManualGenerator(model=model, tokenizer=tokenizer, device='cpu')
    generator.cache_system_prompt('ABC')  # Cache length = 3
    model.calls.clear()  # Drop cache call
    return generator, model


def test_position_ids_with_cached_system_prompt(generator_with_cache):
    generator, model = generator_with_cache

    result = generator.generate('DE', max_new_tokens=3, do_sample=False)
    assert result['num_tokens'] == 3

    assert len(model.calls) == 3

    first_call = model.calls[0]
    torch.testing.assert_close(first_call['position_ids'], torch.tensor([[3, 4]]))
    assert first_call['attention_mask'].shape[1] == 5

    second_call = model.calls[1]
    torch.testing.assert_close(second_call['position_ids'], torch.tensor([[5]]))
    assert second_call['attention_mask'].shape[1] == 6

    third_call = model.calls[2]
    torch.testing.assert_close(third_call['position_ids'], torch.tensor([[6]]))
    assert third_call['attention_mask'].shape[1] == 7


def test_position_ids_without_cache():
    tokenizer = DummyTokenizer()
    model = DummyModel()
    generator = ManualGenerator(model=model, tokenizer=tokenizer, device='cpu')

    result = generator.generate('XY', max_new_tokens=2, do_sample=False, use_cache=False)
    assert result['num_tokens'] == 2

    first_call = model.calls[0]
    torch.testing.assert_close(first_call['position_ids'], torch.tensor([[0, 1]]))

    second_call = model.calls[1]
    torch.testing.assert_close(second_call['position_ids'], torch.tensor([[2]]))


def test_position_ids_with_past_kv():
    tokenizer = DummyTokenizer()
    model = DummyModel()
    generator = ManualGenerator(model=model, tokenizer=tokenizer, device='cpu')

    past = DummyModel.make_past(cache_length=5)
    result = generator.generate('HI', max_new_tokens=2, do_sample=False, past_key_values=past)
    assert result['num_tokens'] == 2

    first_call = model.calls[0]
    torch.testing.assert_close(first_call['position_ids'], torch.tensor([[5, 6]]))

    second_call = model.calls[1]
    torch.testing.assert_close(second_call['position_ids'], torch.tensor([[7]]))
