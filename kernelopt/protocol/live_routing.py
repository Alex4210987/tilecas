"""CUDA paper route contract. Pure, replayable checks; no destination policy."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


VERSION = "tilecas-live-v11-opportunity-stop"
LEVELS = ("High", "Native")


@dataclass
class Reassessment:
    """Use AKO's existing three-stall rule, not a call after every edit."""
    best_ms: float | None = None
    stalled: int = 0
    failed: int = 0
    seen: set = field(default_factory=set)

    def observe(self, checkpoint):
        identity = checkpoint["commit"]
        if identity in self.seen:
            return None
        self.seen.add(identity)
        metrics = checkpoint.get("metrics", {})
        ms = metrics.get("candidate_median_ms") if metrics.get("correctness") else None
        significant = ms is not None and (self.best_ms is None or ms <= self.best_ms * 0.97)
        if ms is not None:
            self.best_ms = min(ms, self.best_ms) if self.best_ms is not None else ms
        if not re.search(r"\[iter\s*\d+\]", checkpoint.get("message", ""), re.I):
            return None
        if ms is None:
            self.failed += 1
            if self.failed >= 3:
                self.failed = 0
                return "correctness_reassessment"
            return None
        self.failed = 0
        self.stalled = 0 if significant else self.stalled + 1
        if self.stalled >= 3:
            self.stalled = 0
            return "ako_stall"
        return None


RELATIONS = {"lowers-to", "realizes", "guards", "depends-on"}
CONTROL_CAUSES = {
    "tiling": {"masked_work", "decomposition", "memory_layout"},
    "fusion": {"decomposition", "transfer_wait"},
    "decomposition": {"decomposition", "masked_work", "repeated_lowering"},
    "layout": {"memory_layout", "masked_work"},
    "pipeline_depth": {"transfer_wait", "resource_pressure"},
    "indexing": {"numerical_mismatch", "masked_work", "compile_error"},
    "precision": {"numerical_mismatch", "instruction_schedule", "compile_error"},
    "synchronization": {"numerical_mismatch", "transfer_wait", "compile_error"},
    "instruction_order": {"instruction_schedule", "repeated_lowering", "resource_pressure"},
    "buffer_lifetime": {"transfer_wait", "resource_pressure"},
    "memory_transfer": {"transfer_wait", "memory_layout"},
    "register_allocation": {"resource_pressure", "instruction_schedule"},
    "shared_memory_layout": {"memory_layout", "resource_pressure"},
    "launch_geometry": {"masked_work", "resource_pressure"},
    "repair": {"compile_error", "numerical_mismatch"},
}
CONTROLS = {
    # T.Kernel(..., threads=...) exposes launch geometry at the source level.
    "High": ["tiling", "fusion", "decomposition", "layout", "pipeline_depth", "launch_geometry", "indexing", "precision", "repair"],
    "Native": ["indexing", "precision", "synchronization", "instruction_order", "buffer_lifetime",
               "memory_transfer", "register_allocation", "shared_memory_layout", "launch_geometry", "repair"],
}
ACTION_FIELDS = {"route", "target_levels", "edit_op", "args", "finding_id", "expected_metric", "stop_if"}


def digest(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).decode("utf-8", errors="surrogateescape")


def parse_action(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != ACTION_FIELDS:
        raise ValueError("action requires exactly the seven paper fields")
    if value["route"] not in {"stay", "switch", "verify", "stop"}:
        raise ValueError("unknown route")
    targets = value["target_levels"]
    if not isinstance(targets, list) or any(x not in LEVELS for x in targets) or len(set(targets)) != len(targets):
        raise ValueError("target_levels must be an ordered list of distinct editable levels")
    if not isinstance(value["args"], dict) or set(value["args"]) != {"object_id", "description"}:
        raise ValueError("args must contain object_id and a declarative description; commands are not executable controls")
    if not all(isinstance(value[k], str) for k in ("edit_op", "finding_id", "expected_metric", "stop_if")):
        raise ValueError("action text fields must be strings")
    if not all(isinstance(v, str) for v in value["args"].values()):
        raise ValueError("control arguments must be strings")
    return value


ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": sorted(ACTION_FIELDS),
    "properties": {
        "route": {"type": "string", "enum": ["stay", "switch", "verify", "stop"]},
        "target_levels": {"type": "array", "items": {"type": "string", "enum": list(LEVELS)}},
        "edit_op": {"type": "string"},
        "args": {"type": "object", "additionalProperties": False,
                 "required": ["object_id", "description"],
                 "properties": {"object_id": {"type": "string"}, "description": {"type": "string"}}},
        "finding_id": {"type": "string"}, "expected_metric": {"type": "string"}, "stop_if": {"type": "string"},
    },
}


def finding_schema(context):
    """Expose the actual wire enums and evidence identities to the policy."""
    def obj(properties):
        return {"type": "object", "additionalProperties": False,
                "required": list(properties), "properties": properties}

    def enum(values):
        return {"enum": list(values)}

    objects = [o["id"] for o in context["objects"]]
    evidence = [obj({"source": {"const": e["observation"]["source"]}, "id": {"const": e["id"]},
                     "relation": enum(sorted(RELATIONS | {"observes"})), "freshness": {"const": "fresh"}})
                for e in context["evidence"]]
    return obj({
        "checkpoint": {"const": context["checkpoint"]},
        "observation": enum([e["observation"] for e in context["evidence"]]),
        "cause": obj({"class": enum(sorted(set.union(*CONTROL_CAUSES.values()))),
                      "object_id": enum(objects), "relation": enum(sorted(RELATIONS | {"observes"}))}),
        "controls": {"type": "array", "items": obj({"level": enum(LEVELS),
            "name": enum(CONTROL_CAUSES), "args": obj({"object_id": enum(objects)}),
            "preconditions": {"const": ["fresh_evidence", "mapped_object", "task_abi"]}})},
        "evidence": {"type": "array", "minItems": 1, "items": {"oneOf": evidence}},
        "action": obj({"levels": {"type": "array", "items": enum(LEVELS)},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                       "rationale": {"type": "string"}}),
        "status": enum(["supported", "verify", "unresolved"]),
    })


@dataclass
class Budget:
    deadline: float
    # Retained for reading historical manifests; current runs impose neither quota.
    bootstrap_limit: int | None = None
    verification_limit: int | None = None
    evaluations: int = 0
    verifications: int = 0
    model_calls: int = 0
    tokens: dict = field(default_factory=dict)
    correct_seen: bool = False
    # A run that has produced nothing correct is not a slow run, it is a
    # stuck one, and it will not become comparable by being given the rest
    # of the envelope. Wall time is the only budget; iteration counts are
    # telemetry, because the scaffolding keys on committed measurements
    # rather than on how many of them there are.
    first_correct_deadline: float | None = None

    def available(self, now, seconds=0):
        # Fixed checks wall time while an agent is active, without reserving
        # projected export, replay or endpoint time. Counts are telemetry only.
        return self.expiry(now) is None

    def expiry(self, now):
        """Why this run must end now, or None while it may continue."""
        if now >= self.deadline:
            return "agent wall-time limit exhausted"
        if (self.first_correct_deadline is not None and not self.correct_seen
                and now >= self.first_correct_deadline):
            return "no correct candidate within the first-correct window"
        return None

    def record_evaluation(self, correct):
        self.evaluations += 1
        self.correct_seen |= bool(correct)

    def view(self, now):
        return {**self.__dict__, "seconds_remaining": max(0, self.deadline - now)}


def map_finding(finding, context):
    """Resolve evidence first, then traverse only authenticated graph edges."""
    if not isinstance(finding, dict) or set(finding) != {
        "checkpoint", "observation", "cause", "controls", "evidence", "action", "status"
    }:
        return {}, "verify", "invalid typed finding"
    if finding["checkpoint"] != context["checkpoint"]:
        return {}, "verify", "stale checkpoint"
    if finding["status"] not in {"supported", "verify", "unresolved"}:
        return {}, "verify", "invalid finding status: expected supported, verify, or unresolved"
    observation = finding["observation"]
    if observation not in [e["observation"] for e in context["evidence"]]:
        return {}, "verify", "observation is not an acquired measurement"
    evidence = {e["id"]: e for e in context["evidence"]}
    if not finding["evidence"]:
        return {}, "verify", "finding has no evidence"
    cited = []
    for ref in finding["evidence"]:
        e = evidence.get(ref.get("id"))
        if not e:
            return {}, "verify", f"unknown evidence id: {ref.get('id')}"
        if e["checkpoint"] != context["checkpoint"] or e["observed_at"] < context["created_at"]:
            return {}, "verify", f"stale evidence: {e['id']}"
        if ref.get("source") != e["observation"]["source"]:
            return {}, "verify", f"evidence {e['id']}: source must equal {e['observation']['source']!r}; an object_id is not a source"
        if ref.get("freshness") != "fresh":
            return {}, "verify", "evidence freshness must be 'fresh'"
        if ref.get("relation") not in RELATIONS | {"observes"}:
            return {}, "verify", "unknown evidence relation"
        cited.append(e)
    objects = {o["id"]: o for o in context["objects"]}
    start = objects.get(observation.get("object_id"))
    cause = finding["cause"]
    if not start or start["grade"] == "associated" or cause.get("relation") not in RELATIONS | {"observes"}:
        return {}, "verify", "observation lacks exact or compiler provenance"
    # Current profiler records bind a launch observation to the measured whole
    # program, not to one generated callsite, IR function, or native kernel.
    # Restrict only when that profile record is the finding's selected
    # observation. Other cited measurements may be supporting context for a
    # latency-led structural hypothesis and must not silently change its scope.
    selected_profile = [record for record in cited
                        if record.get("observation") == observation
                        and observation.get("kind") == "profile"]
    program_only_profile = any(
        "source_mapping" in record or "measurement_identity" in record
        for record in selected_profile
    )
    if program_only_profile and not str(cause.get("object_id", "")).endswith(":program"):
        return {}, "unresolved", "profile launch is bound only to the whole program, not a narrow source object"
    reached = {start["id"]: start["grade"]}
    changed = True
    while changed:
        changed = False
        for edge in context["relations"]:
            if edge["relation"] not in RELATIONS or edge.get("checkpoint") != context["checkpoint"]:
                continue
            if not edge.get("proof") or edge.get("grade") == "associated":
                continue
            a, b = edge["from"], edge["to"]
            if a in reached and b not in reached and b in objects:
                reached[b] = "mapped"
                changed = True
            if b in reached and a not in reached and a in objects:
                reached[a] = "mapped"
                changed = True
    if cause.get("object_id") not in reached:
        return {}, "verify", "cause object is unmapped"
    mapped = {}
    for proposal in finding["controls"]:
        level, name = proposal.get("level"), proposal.get("name")
        if name not in CONTROLS.get(level, ()) or cause.get("class") not in CONTROL_CAUSES.get(name, ()):
            continue
        if proposal.get("preconditions") != ["fresh_evidence", "mapped_object", "task_abi"]:
            continue
        object_id = proposal.get("args", {}).get("object_id")
        if object_id not in reached or objects[object_id]["level"] != level:
            continue
        if program_only_profile and not str(object_id).endswith(":program"):
            continue
        if not context["targets"].get(level, {}).get("abi_ok"):
            continue
        mapped.setdefault(level, []).append({"name": name, "object_id": object_id, "grade": reached[object_id]})
    return (mapped, "supported", "mapped declared controls") if mapped else ({}, "unresolved", "no mapped corrective control")


def gate(action, finding, context, budget, now, switched):
    """Return first *proposed* legal target. Does not mutate branch or budget."""
    a = parse_action(action)
    reject = lambda reason: {"accepted": False, "reason": reason, "target": None}
    if not budget.available(now):
        return reject("agent wall-time limit exhausted")
    if a["finding_id"] != context["finding_id"]:
        return reject("stale finding id")
    mapped, status, reason = map_finding(finding, context)
    if a["route"] == "stop":
        if a["edit_op"] != "stop" or a["target_levels"]:
            return reject("invalid stop action")
        if not a["stop_if"].strip():
            return reject("stop requires an explicit natural or value rationale")
        return ({"accepted": True, "target": None,
                 "reason": "validated incumbent permits endpoint finalization",
                 "mapping": mapped}
                if budget.correct_seen else reject("stop requires a validated incumbent"))
    if a["route"] == "verify":
        if a["target_levels"] != [context["level"]]:
            return reject("verify must name the active level")
        if a["edit_op"] not in {"profile", "mapping", "replay"}:
            return reject("unknown verification operation")
        return {"accepted": True, "target": context["level"], "reason": "requested verification", "mapping": mapped}
    if status != "supported":
        return reject(status + ": " + reason)
    if a["route"] == "stay" and a["target_levels"] != [context["level"]]:
        return reject("stay must name only the active level")
    reasons = []
    for level in a["target_levels"]:
        target = context["targets"].get(level, {})
        control = next((c for c in mapped.get(level, ()) if c["name"] == a["edit_op"]), None)
        # Fresh route-call IDs are not new evidence. Preserve the duplicate
        # denominator until observations or checkpoint ancestry really change.
        identity = {"observation": finding["observation"], "cause": finding["cause"],
                    "controls": finding["controls"], "acquired": [e["observed_at"] for e in context["evidence"]]}
        key = (digest(identity), context["ancestry"], level)
        if not control:
            reasons.append(f"{level}: control is unavailable or unmapped")
        elif a["args"]["object_id"] != cause_object(finding) and a["args"]["object_id"] != control["object_id"]:
            reasons.append(f"{level}: action object is unmapped")
        elif not target.get("reachable") or not target.get("materializable") or not target.get("abi_ok"):
            reasons.append(f"{level}: unreachable, incompatible ABI, or no materializable checkpoint")
        elif level != context["level"] and (not context["correct"] or not target.get("correctness_guard")):
            reasons.append(f"{level}: transition correctness guard failed")
        elif level != context["level"] and key in switched:
            reasons.append(f"{level}: duplicate finding and checkpoint ancestry")
        else:
            return {"accepted": True, "target": level, "reason": "first admissible proposed target",
                    "mapping": mapped, "filtered": reasons, "switch_key": key, "control": control}
    return reject("; ".join(reasons) or "no proposed target")


def cause_object(finding):
    return finding.get("cause", {}).get("object_id")
