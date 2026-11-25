"""Execute a portable action graph with the same history and action codecs as PT."""

import onnxruntime as ort
import torch

from sequifier.io.action_record import action_record_layout
from sequifier.io.actions import action_record, decode_action
from sequifier.model.execution_schema import policy_graph_schema
from sequifier.rl.attachment import bind_policy
from sequifier.rl.policy import ExecutablePolicy


class OnnxPolicy(ExecutablePolicy):
    def __init__(self, model, device):
        super().__init__()
        self.execution_config = model.execution
        self.interface_name = model.interface_name
        self.config = bind_policy(self.execution_config, self.interface_name)
        # Histories and seeded draws are host tensors. ORT owns device transfers.
        self.device = torch.device("cpu")
        execution_device = torch.device(device)
        self.schema, expected = policy_graph_schema(self.interface, self.config)
        if expected != model.onnx_schema:
            raise ValueError("ONNX policy schema differs from its action binding")
        provider = (
            "CUDAExecutionProvider"
            if execution_device.type == "cuda"
            else "CPUExecutionProvider"
        )
        if provider not in ort.get_available_providers():
            raise ValueError(f"Requested ONNX provider is unavailable: {provider}")
        options = (
            {"device_id": execution_device.index or 0}
            if execution_device.type == "cuda"
            else {}
        )
        self.session = ort.InferenceSession(model.path, providers=[(provider, options)])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.eval()

    def _select(self, inputs, deterministic, generator, input_encoding):
        features, metadata = self.model_inputs(inputs, input_encoding=input_encoding)
        batch = inputs.attention_valid_mask.shape[0]
        count = len(self.config.actions)
        draws = torch.zeros((batch, 1 + 2 * count), dtype=torch.float64)
        if not deterministic:
            draws[:, 0] = 1
            draws[:, 1::2] = torch.rand(
                (batch, count),
                generator=generator,
                device=generator.device,
                dtype=torch.float64,
            ).cpu()
            draws[:, 2::2] = torch.randn(
                (batch, count),
                generator=generator,
                device=generator.device,
                dtype=torch.float64,
            ).cpu()
        feeds = (
            {"policy_draws": draws.numpy()}
            if "policy_draws" in self.input_names
            else {}
        )
        for descriptor in self.schema.inputs:
            if descriptor.name in self.input_names:
                tensors = metadata if descriptor.role == "metadata" else features
                feeds[descriptor.name] = tensors[descriptor.key].detach().cpu().numpy()
        values = self.session.run(None, feeds)
        outputs = {
            item.name: torch.from_numpy(value)
            for item, value in zip(self.session.get_outputs(), values, strict=True)
        }
        if any(
            not value.isfinite().all()
            for value in outputs.values()
            if value.is_floating_point()
        ):
            raise ValueError(
                "ONNX policy returned non-finite decisions or probabilities"
            )
        actions, probabilities, latents, components = {}, {}, {}, {}
        for index, (name, field) in enumerate(self.config.actions.items()):
            actions[name] = outputs[f"action_{index}"]
            if field.type == "categorical":
                probabilities[name] = outputs[f"probabilities_{index}"]
            else:
                latents[name] = outputs[f"latent_{index}"]
                components[name] = outputs[f"component_{index}"]
        record = action_record_layout(self.config).pack(actions, latents, components)
        return record, outputs["joint_log_probability"], probabilities

    def predict_actions(self, inputs, deterministic, generator, *, probabilities):
        record, scores, conditional = self._select(
            inputs, deterministic, generator, "model"
        )
        actions = [
            decode_action(action_record(record, index), self.config)
            for index in range(scores.shape[0])
        ]
        return (
            actions,
            scores if probabilities else None,
            conditional if probabilities else {},
        )

    def action_scores(self, inputs, *, deterministic, generator, entropy):
        if entropy:
            raise ValueError("The ONNX action graph does not output entropy")
        record, scores, _ = self._select(inputs, deterministic, generator, None)
        return record, scores, None
