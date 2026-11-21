"""Own reference resolution, validation and frozen weight reuse."""

from copy import deepcopy
from pathlib import Path

from sequifier.artifacts.state_dict import weight_fingerprint
from sequifier.fingerprints import file_checksum
from sequifier.rl.checkpoint import load_policy
from sequifier.rl.run_state import require_equal


class PolicyReferences:
    def __init__(self, config, recovery, *, resuming):
        self.config, self.recovery = config, recovery
        self.artifacts, self.checksums = {}, {}
        if (
            resuming
            and config.initialize_from
            and Path(config.initialize_from).is_file()
        ):
            if config.actor_initialization == "history_only":
                require_equal(
                    recovery.initialization["transfer"]["source_sha256"],
                    file_checksum(config.initialize_from),
                    "resume.initialize_from.sha256",
                )
            else:
                _, identity = self._artifact(config.initialize_from)
                require_equal(
                    recovery.initialization["weight_hash"],
                    identity,
                    "resume.initialize_from.weights",
                )
        for index, phase in enumerate(config.phases):
            reference = phase.reference_config
            if reference is None or reference.source != "artifact":
                continue
            if Path(reference.path).is_file():
                _, identity = self._artifact(reference.path)
                expected = recovery.reference_contract.setdefault(
                    str(config.phase_offset + index), identity
                )
                require_equal(
                    expected, identity, f"phases.{phase.name}.reference.weights"
                )
            elif index > recovery.phase_index or (
                index == recovery.phase_index and recovery.phase_state is None
            ):
                raise FileNotFoundError(
                    f"reference artifact required before phase start: {reference.path}"
                )
        # Completed references and initialization are no longer needed in memory.
        pending = {
            reference.path
            for phase in config.phases[recovery.phase_index :]
            if (reference := phase.reference_config) is not None
            and reference.source == "artifact"
        }
        self.artifacts = {
            path: value for path, value in self.artifacts.items() if path in pending
        }

    def _artifact(self, path):
        checksum = file_checksum(path)
        if path not in self.artifacts or self.checksums[path] != checksum:
            source = load_policy(
                path,
                device="cpu",
                interface_name=self.config.policy.interface_name,
                binding=self.config.policy,
                model=self.config.parent.execution.model,
            )
            self.recovery.representation.validate_actor(source, f"reference.{path}")
            weights = source.network.state_dict()
            self.artifacts[path] = weights, weight_fingerprint(weights)
            self.checksums[path] = checksum
        return self.artifacts[path]

    def resolve(self, actor, index):
        recovery = self.recovery
        declaration = self.config.phases[recovery.phase_index].reference_config
        if declaration is None:
            return None, None
        if recovery.phase_state is not None:
            weights, identity = (
                recovery.phase_state["reference"],
                recovery.phase_state["reference_identity"],
            )
        elif declaration.source == "artifact":
            weights, identity = self._artifact(declaration.path)
            require_equal(
                recovery.reference_contract[str(index)],
                identity,
                "reference.artifact.weights",
            )
        else:
            weights = recovery.incoming_actor
            identity = weight_fingerprint(weights)
        reference = deepcopy(actor).eval().requires_grad_(False)
        reference.network.load_state_dict(weights, strict=True)
        return reference, identity


def validate_reference_state(phase, recovery, index):
    state = recovery.phase_state
    if state is not None:
        reference = phase.reference_config
        if (reference is None) != (state["reference"] is None):
            raise ValueError("checkpoint reference disagrees with the phase")
        if reference is not None:
            identity = weight_fingerprint(state["reference"])
            require_equal(
                identity, state["reference_identity"], "resume.embedded_reference"
            )
            expected = (
                weight_fingerprint(recovery.incoming_actor)
                if reference.source == "phase_start"
                else recovery.reference_contract[str(index)]
            )
            require_equal(expected, identity, "resume.reference.weights")
