"""Development-only static-report outcome routing baseline.

Full-mission static outcomes are proxies for future adaptive outcomes, not
certificates. Altitude context is public; report correctness is never observed.
"""

from experiments.v4_benchmark.calibrated_router import CalibratedRouter


class ReportCalibratedRouter(CalibratedRouter):
    def validate_protocol(self, scenario):
        protocol = self.artifact.get("report_protocol")
        if protocol is None:
            raise ValueError("missing report calibration")
        for key, value in dict(
            reporting=scenario.reporting.to_dict(),
            duration_s=scenario.duration_s,
            interval_s=scenario.observation_interval_s,
            resources=scenario.resources.to_dict(),
            execution_costs={
                spec.config_id: [spec.mission_latency_s, spec.energy_j_per_call]
                for spec in scenario.executors
            },
        ).items():
            if protocol[key] != value:
                raise ValueError(f"report calibration protocol mismatch: {key}")
        if self.interval_s != protocol["interval_s"]:
            raise ValueError("report calibration cadence mismatch")

    def select(self, observation):
        super().select(observation)
        context = "below_6m" if observation.position_m[2] < 6 else "at_least_6m"
        req = observation.requirements
        candidates = self.last_decision["candidates"]
        for option in candidates:
            stats = self.artifact["report_outcomes"][context][option["config_id"]]
            checks = {}
            for field, metric in (
                ("min_report_recall", "report_recall"),
                ("max_false_reports", "false_reports"),
                ("max_duplicate_reports", "duplicate_correct_reports"),
            ):
                bound = getattr(req, field)
                if bound is not None:
                    checks[field] = (
                        stats[metric] >= bound
                        if field.startswith("min_")
                        else stats[metric] <= bound
                    )
            if (
                req.confirmed_report_deadline_s is not None
                and observation.confirmed_reports_so_far == 0
            ):
                first = stats["first_confirmed_correct_report_s"]
                checks["confirmed_report_deadline_s"] = (
                    first is not None
                    and observation.state.current_time_s + first + option["model_load_latency_s"]
                    <= req.confirmed_report_deadline_s
                )
            option.update(
                report_proxy_checks=checks,
                report_proxy=stats,
                report_proxy_meets_request=all(checks.values()),
            )
        feasible = [
            c for c in candidates if c["memory_fits"] and c["deadline_fits"] and c["battery_fits"]
        ]
        qualified = [
            c for c in feasible if c["proxy_meets_request"] and c["report_proxy_meets_request"]
        ]
        if qualified:
            chosen = min(
                qualified,
                key=lambda c: (c["projected_remaining_energy_j"], c["latency_s"], c["config_id"]),
            )
            reason = "lowest_projected_energy_meeting_development_report_proxy"
        else:
            pool = feasible or [c for c in candidates if c["memory_fits"]] or candidates
            chosen = max(
                pool,
                key=lambda c: (
                    sum(c["report_proxy_checks"].values()),
                    c["report_proxy"]["report_recall"],
                    -c["report_proxy"]["false_reports"],
                    -c["report_proxy"]["duplicate_correct_reports"],
                    -c["projected_remaining_energy_j"],
                ),
            )
            reason = "report_request_infeasible_best_effort"
        self.last_decision.update(
            selected_config_id=chosen["config_id"],
            reason=reason,
            report_requirements_consumed=True,
            report_context=context,
            report_proxy_scope=(
                "development static full-mission outcomes; adaptive results not guaranteed"
            ),
            report_deadline_guard="until first unverified confirmed report",
        )
        self.last_decision.pop("report_requirements_note", None)
        return chosen["config_id"]
