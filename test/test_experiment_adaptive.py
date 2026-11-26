"""Regression tests for feedback-dependent acquisition through the core runner."""

import time
from unittest.mock import patch

from artiq.language import delay, kernel, now_mu, rpc
from artiq.language.core import TerminationRequested
from artiq.language.types import TFloat, TInt32, TTuple
from emulator_environment import KernelEmulatorCase
from mock_environment import MockScheduler

from ndscan.experiment.entry_point import make_fragment_scan_exp
from ndscan.experiment.fragment import (
    ExpFragment,
    RestartKernelTransitoryError,
    TransitoryError,
)
from ndscan.experiment.optimizers.coordinate_search import CoordinateSearchOptimizer
from ndscan.experiment.parameters import FloatParam
from ndscan.experiment.result_channels import FloatChannel


class FeedbackFragment(ExpFragment):
    def build_fragment(self):
        self.setattr_device("core")
        self.setattr_device("scheduler")
        self.setattr_param("x", FloatParam, "x", 0.0)
        self.setattr_param("y", FloatParam, "y", 0.0)
        self.setattr_result("objective", FloatChannel)
        self.setattr_result("host_x", FloatChannel)
        self.attempts = []
        self.faults = {}
        self.advance_time = False
        self.pause_after_attempt = None
        self.host_setup_count = 0
        self.host_cleanup_count = 0
        self.device_cleanup_count = 0

    def device_cleanup(self):
        self.device_cleanup_count += 1

    def host_setup(self):
        self.host_setup_count += 1

    def host_cleanup(self):
        self.host_cleanup_count += 1

    @rpc
    def measure(self, x: TFloat, y: TFloat) -> TTuple([TFloat, TInt32]):
        # Check the host and core parameter stores agree at every acquisition,
        # including after a chunk boundary or a kernel restart.
        assert x == self.x.get()
        assert y == self.y.get()
        attempt = len(self.attempts)
        self.attempts.append((x, y))
        if attempt == self.pause_after_attempt:
            self.scheduler.pause_requested = True
        # Add a deterministic drift/shot-dependent contribution, to expose
        # reordering or duplicate acquisitions in addition to coordinate errors.
        value = (x - 1.25) ** 2 + (y + 0.5) ** 2 + 2.0 + attempt / 100
        return value, self.faults.get(attempt, 0)

    @kernel
    def run_once(self):
        if self.advance_time:
            self.core.break_realtime()
            delay(0.25)
            self.core.wait_until_mu(now_mu())
        value, fault = self.measure(self.x.get(), self.y.get())
        # A partially published result must be discarded on retry or skip.
        self.objective.push(value)
        if fault == 1:
            raise TransitoryError
        if fault == 2:
            raise RestartKernelTransitoryError
        self.host_x.push(self.x.get())


FeedbackScan = make_fragment_scan_exp(FeedbackFragment, max_transitory_error_retries=1)


class PausingScheduler(MockScheduler):
    def __init__(self):
        super().__init__()
        self.pause_requested = False
        self.pauses = 0
        self.terminate_on_pause = False

    def check_pause(self) -> bool:
        return self.pause_requested or super().check_pause()

    def pause(self):
        self.pauses += 1
        self.pause_requested = False
        if self.terminate_on_pause:
            raise TerminationRequested
        super().pause()


class AdaptiveKernelCase(KernelEmulatorCase):
    def run_case(
        self,
        *,
        algorithm="coordinate_search",
        budget=12,
        repeats=1,
        reference="none",
        interval=1,
        averaging="mean",
        direction="min",
        faults=None,
        pause_after_attempt=None,
        terminate_on_pause=False,
        ask_delay=0.0,
    ):
        scheduler = PausingScheduler()
        scheduler.terminate_on_pause = terminate_on_pause
        self.device_mgr.virtual_devices["scheduler"] = scheduler
        exp = self.create(FeedbackScan)
        fragment = exp.fragment
        fragment.faults = faults or {}
        fragment.advance_time = pause_after_attempt is not None
        fragment.pause_after_attempt = pause_after_attempt
        exp.args._params["execution_mode"] = "optimise"
        algorithm_spec = {
            "kind": algorithm,
            "xatol": 1e-8,
            "fatol": 1e-9,
        }
        if algorithm in ("nelder_mead", "bayesian"):
            algorithm_spec["user_seed"] = 7
        if algorithm == "bayesian":
            algorithm_spec["n_init"] = 10
        exp.args._params["optimise"] = {
            "parameters": [
                {
                    "fqn": f"{__name__}.FeedbackFragment.{name}",
                    "path": "*",
                    "min": -5.0,
                    "max": 5.0,
                    "initial": initial,
                }
                for name, initial in (("x", 4.0), ("y", -4.0))
            ],
            "objective": {"channel": "objective", "direction": direction},
            "algorithm": algorithm_spec,
            "max_evals": budget,
            "num_repeats_per_point": repeats,
            "reference_normalisation": reference,
            "reference_resample_interval": interval,
            "averaging_method": averaging,
            "skip_on_persistent_transitory_error": True,
        }
        exp.prepare()
        objective_sink = fragment.objective.sink
        core = self.device_mgr.get("core")
        compile_functions = []
        compile_seconds = []
        real_compile = core.compile

        def record_compile(function, *args, **kwargs):
            compile_functions.append(function.__name__)
            start = time.perf_counter()
            try:
                return real_compile(function, *args, **kwargs)
            finally:
                compile_seconds.append(time.perf_counter() - start)

        real_ask = CoordinateSearchOptimizer.ask
        delayed_asks = []

        def delayed_ask(optimizer):
            if fragment.attempts:
                delayed_asks.append(len(fragment.attempts))
                time.sleep(ask_delay)
            return real_ask(optimizer)

        start = time.perf_counter()
        with patch.object(core, "compile", side_effect=record_compile):
            if ask_delay:
                assert algorithm == "coordinate_search"
                with patch.object(CoordinateSearchOptimizer, "ask", delayed_ask):
                    exp.run()
            else:
                exp.run()
        wall_seconds = time.perf_counter() - start
        self.assertIs(fragment.objective.sink, objective_sink)
        self.assertEqual(fragment.device_cleanup_count, 1)

        prefix = "ndscan.rid_0."
        datasets = {
            key[len(prefix) :]: self.dataset_db.get(key)
            for key in self.dataset_db.data
            if key.startswith(prefix)
            and (
                key[len(prefix) :].startswith(("points.", "optimizer."))
                or key == prefix + "completed"
            )
        }
        if terminate_on_pause:
            self.assertFalse(datasets.get("completed", False))
        else:
            self.assertTrue(datasets["completed"])
        self.assertEqual(
            datasets.get("points.axis_0", []), datasets.get("points.channel_host_x", [])
        )
        return {
            "datasets": datasets,
            "attempts": fragment.attempts,
            "compiles": compile_functions,
            "compile_seconds": compile_seconds,
            "wall_seconds": wall_seconds,
            "pauses": scheduler.pauses,
            "host_setup_count": fragment.host_setup_count,
            "host_cleanup_count": fragment.host_cleanup_count,
            "device_cleanup_count": fragment.device_cleanup_count,
            "delayed_asks": delayed_asks,
        }

    def assert_acquisitions(self, result, raw, evaluations, compiles):
        datasets = result["datasets"]
        self.assertEqual(len(datasets.get("points.axis_0", [])), raw)
        self.assertEqual(
            len(datasets.get("optimizer.evaluations.objective", [])), evaluations
        )
        self.assertEqual(len(result["compiles"]), compiles, result["compiles"])
        self.assertEqual(result["host_setup_count"], result["host_cleanup_count"])

    def test_multi_candidate_single_compile(self):
        r = self.run_case()
        self.assert_acquisitions(r, 12, 12, 1)

    def test_nelder_mead_single_compile(self):
        r = self.run_case(algorithm="nelder_mead")
        self.assert_acquisitions(r, 12, 12, 1)

    def test_bayesian_single_compile(self):
        # n_init is now per dimension: 20 initial samples in 2D, then
        # four model-driven requests.
        r = self.run_case(algorithm="bayesian", budget=24)
        self.assert_acquisitions(r, 24, 24, 1)

    def test_repeats_span_chunk_boundaries(self):
        r = self.run_case(budget=39, repeats=13, averaging="median")
        self.assert_acquisitions(r, 39, 3, 1)
        self.assertEqual(
            r["datasets"]["optimizer.evaluations.point_index"], [12, 25, 38]
        )

    def test_subtract_reference_repeats(self):
        r = self.run_case(budget=18, repeats=3, reference="subtract")
        self.assert_acquisitions(r, 18, 3, 1)
        self.assertEqual(
            r["datasets"]["optimizer.evaluations.point_index"], [5, 11, 17]
        )

    def test_divide_reference_maximise(self):
        r = self.run_case(budget=18, repeats=3, reference="divide", direction="max")
        self.assert_acquisitions(r, 18, 3, 1)

    def test_retry_and_partial_skip(self):
        # First shot skipped after two failures; second shot retries once.
        r = self.run_case(budget=12, repeats=3, faults={0: 1, 1: 1, 2: 1})
        self.assert_acquisitions(r, 11, 4, 1)
        self.assertEqual(len(r["attempts"]), 14)

    def test_failed_reference_terminates(self):
        r = self.run_case(
            budget=12, repeats=3, reference="subtract", faults={0: 1, 1: 1}
        )
        self.assert_acquisitions(r, 2, 0, 1)
        self.assertEqual(
            r["datasets"]["optimizer.termination_reason"],
            "reference_measurement_failed",
        )

    def test_restart_mid_batch(self):
        r = self.run_case(budget=26, repeats=13, faults={4: 2})
        self.assert_acquisitions(r, 26, 2, 2)
        self.assertEqual(r["pauses"], 1)
        self.assertEqual(len(r["attempts"]), 27)

    def test_scheduler_pause_mid_batch(self):
        r = self.run_case(budget=6, repeats=3, pause_after_attempt=0)
        self.assert_acquisitions(r, 6, 2, 2)
        self.assertEqual(r["pauses"], 1)

    def test_slow_optimizer_rpc(self):
        r = self.run_case(budget=12, repeats=3, ask_delay=0.5)
        self.assert_acquisitions(r, 12, 4, 1)
        self.assertEqual(r["delayed_asks"], [3, 6, 9])

    def test_terminate_during_persistent_acquisition(self):
        r = self.run_case(
            budget=30, repeats=3, pause_after_attempt=3, terminate_on_pause=True
        )
        self.assertEqual(r["pauses"], 1)
        self.assertEqual(len(r["compiles"]), 1)
        self.assertGreaterEqual(len(r["attempts"]), 4)
        self.assertLess(len(r["attempts"]), 30)
        self.assertEqual(r["host_setup_count"], r["host_cleanup_count"])
        self.assertEqual(r["device_cleanup_count"], 1)

    def test_all_skipped_candidate(self):
        r = self.run_case(budget=9, repeats=3, faults=dict.fromkeys(range(6), 1))
        self.assert_acquisitions(r, 6, 3, 1)
        self.assertEqual(len(r["attempts"]), 12)

    def test_non_multiple_budget_preserves_repeat_group(self):
        r = self.run_case(budget=10, repeats=3)
        self.assert_acquisitions(r, 12, 4, 1)
        self.assertEqual(
            r["datasets"]["optimizer.termination_reason"], "max_evals_reached"
        )
