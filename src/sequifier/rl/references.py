"""Own reference resolution, validation and frozen weight reuse."""

from copy import deepcopy
from pathlib import Path

from sequifier.rl.checkpoint import load_policy
from sequifier.rl.run_state import require_equal, weight_fingerprint
from sequifier.trajectory.sources import file_checksum


class PolicyReferences:
    def __init__(self, config, recovery, *, resuming):
        self.config, self.recovery = config, recovery
        self.artifacts, self.checksums = {}, {}
        if (
            resuming
            and config.initialize_from
            and Path(config.initialize_from).is_file()
        ):
            _, identity = self._artifact(config.initialize_from)
            require_equal(
                recovery.initialization["weight_hash"],
                identity,
                "resume.initialize_from.weights",
            )
        for index, phase in enumerate(config.phases):
            reference = getattr(phase, "reference", None)
            if reference is None or reference.source != "artifact":
                continue
            if Path(reference.path).is_file():
                _, identity = self._artifact(reference.path)
                expected = recovery.reference_contract.setdefault(str(index), identity)
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
            phase.reference.path
            for phase in config.phases[recovery.phase_index :]
            if getattr(phase, "reference", None) is not None
            and phase.reference.source == "artifact"
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
                interface_name=self.config.policy_interface,
                binding=self.config.policy if self.config.policy.is_joint else None,
            )
            self.recovery.representation.validate_actor(
                source, self.config.units, f"reference.{path}"
            )
            weights = source.network.state_dict()
            self.artifacts[path] = weights, weight_fingerprint(weights)
            self.checksums[path] = checksum
        return self.artifacts[path]

    def resolve(self, actor):
        recovery = self.recovery
        declaration = getattr(
            self.config.phases[recovery.phase_index], "reference", None
        )
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
                recovery.reference_contract[str(recovery.phase_index)],
                identity,
                "reference.artifact.weights",
            )
        else:
            weights = recovery.incoming_actor
            identity = weight_fingerprint(weights)
        reference = deepcopy(actor).eval().requires_grad_(False)
        reference.network.load_state_dict(weights, strict=True)
        return reference, identity


def validate_reference_state(phase, recovery):
    state = recovery.phase_state
    if state is not None:
        reference = getattr(phase, "reference", None)
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
                else recovery.reference_contract[str(recovery.phase_index)]
            )
            require_equal(expected, identity, "resume.reference.weights")
