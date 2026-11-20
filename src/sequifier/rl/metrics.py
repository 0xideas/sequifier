"""Policy observations in the native supervised metric tables."""

import math
import time

from sequifier.training.metrics import StructuredMetricWriters


class PolicyMetrics:
    def __init__(self, config, identity):
        self.writer = StructuredMetricWriters(
            config.project_root,
            config.model_name,
            0,
            dataset_name=config.policy_interface,
        )
        self.run_id = identity.run_id
        self.session_id = identity.session_id
        self.started = time.monotonic()
        self.pending = None

    def record(
        self,
        session,
        values,
        *,
        validation=False,
        primary=None,
        initial=False,
        final=False,
    ):
        context = session.context
        if not validation:
            self.pending = (session, values)
        if (
            not validation
            and not final
            and not session.stopping.triggered
            and session.transaction
            % context.run.parent.run.logging.log_interval_iterations
        ):
            return
        scalars = {
            name: float(value)
            for name, value in values.items()
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
        }
        if not scalars:
            return
        if primary is None:
            primary = next(
                (
                    name
                    for name in (
                        "training_action_nll",
                        "actor_loss",
                        "training_dpo_loss",
                        "loss",
                        "policy_loss",
                    )
                    if name in scalars
                ),
                next(iter(scalars)),
            )
        self.writer.phase = context.phase.name
        self.writer.algorithm = context.phase.algorithm
        now = time.monotonic()
        self.writer.write_observation(
            validation=validation,
            run_id=self.run_id,
            session_id=self.session_id,
            epoch=context.index + 1,
            iteration=session.transaction,
            batches_total=session.iteration_budget,
            global_step=sum(session.steps.values()),
            values=scalars,
            primary=primary,
            learning_rate=float(session.optimization["actor"][0].param_groups[0]["lr"]),
            elapsed_seconds=now - self.started,
            initial=initial,
        )
        if not validation:
            self.started = now
            self.pending = None

    def flush(self, session):
        if self.pending is not None and self.pending[0] is session:
            self.record(session, self.pending[1], final=True)
