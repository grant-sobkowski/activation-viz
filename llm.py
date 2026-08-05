import json
from dataclasses import asdict, dataclass

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"


def download_model() -> None:
    """Downloads the model into the local huggingface cache, if not already cached."""
    snapshot_download(MODEL_ID)


@dataclass
class ProfiledToken:
    """Represents a single output token.
    Properties:
        text: textual representation of the output token
        tensors: list of tensor weights as 0.00 - 1.00.
    """

    text: str
    tensors: list[list[float]]


@dataclass
class _LayerActivations:
    """Layer activation data for all forward passes ran"""

    llm_layer: int
    tokens: list[list[float]]


class ProfiledSmolLM:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

    def run(self, input_text: str) -> list[ProfiledToken]:
        input_tokens = self._format_input(input_text)

        model_output = self._generate(input_tokens)
        output_tokens: list[str] = self._parse_output_tokens(model_output)

        profiled_tokens: list[ProfiledToken] = []
        for i, token in enumerate(output_tokens):
            # tensors as list of each layer's activations
            token_tensors: list[list[float]] = []

            # Append each layer's tensor corresponding to current token
            for layer_tensors in self.layers.values():
                token_tensors.append(layer_tensors.tokens[i])

            profiled_tokens.append(ProfiledToken(token, token_tensors))

        return profiled_tokens

    def _create_layer_hook(self, layer: int):
        def hook(_module, _input, output):
            del _module, _input  # unused

            # skip adding hooks for prefill
            if output.shape[1] > 1:
                return

            normalized = self.model.model.norm(output.detach())
            self.layers[layer].tokens.append(normalized[0, 0].tolist())

        return hook

    def _format_input(self, input_text: str):
        prompt = [
            {
                "role": "system",
                "content": "Answer concisely, in no more than one paragraph.",
            },
            {"role": "user", "content": input_text},
        ]
        return self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

    def _parse_output_tokens(self, model_output: torch.Tensor) -> list[str]:
        tokens = self.tokenizer.convert_ids_to_tokens(
            model_output, skip_special_tokens=True
        )
        assert isinstance(tokens, list)
        assert all(isinstance(token, str) for token in tokens)

        num_profiled_tokens = len(self.layers[0].tokens)
        num_tokens = len(tokens)

        assert num_tokens > num_profiled_tokens, (
            f"Expected at least {num_profiled_tokens} output tokens, got: {num_tokens}"
        )

        end = num_tokens - 1  # Last output token isn't profiled
        start = end - num_profiled_tokens

        parsed = []
        for pt in tokens[start:end]:
            pt = pt.replace("Ġ", " ").replace("Ċ", "\n")
            parsed.append(pt)
        return parsed

    def _generate(self, input_tokens, max_new_tokens=300) -> torch.Tensor:

        # GUI activation display not dynamic, model must have 30 layers
        num_layers = len(self.model.model.layers)
        assert num_layers == 30, f"Expected 30 layers in model, got: {num_layers}"

        # Reset profiling data (or create it on first time)
        hooks = []
        self.layers: dict[int, _LayerActivations] = {
            i: _LayerActivations(i, []) for i in range(len(self.model.model.layers))
        }
        for i, layer in enumerate(self.model.model.layers):
            hook = layer.mlp.down_proj.register_forward_hook(self._create_layer_hook(i))
            # store references to hooks for cleanup
            hooks.append(hook)

        outputs: torch.Tensor = self.model.generate(**input_tokens, max_new_tokens=300)  # pyright: ignore
        assert isinstance(outputs, torch.Tensor)
        assert len(outputs) == 1
        assert isinstance(outputs[0], torch.Tensor)

        # Cleanup profiling hooks
        for hook in hooks:
            hook.remove()

        return outputs[0]


def main() -> None:
    input_text = "What is the capital of France?"
    model = ProfiledSmolLM()
    output_tokens: list[ProfiledToken] = model.run(input_text)

    from pprint import pprint

    pprint(output_tokens[0:3], depth=2)

    main()
