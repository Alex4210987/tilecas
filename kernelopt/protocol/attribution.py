"""Join measured device tasks to the source objects that produced them.

Correspondence is compiled provenance: the exporter knows which High call site
became which generated kernel.  Measurement is a flat list of device tasks with
durations.  Neither alone answers the only question a cross-level optimizer
actually asks -- *which part of the program I am allowed to edit owns this
cost* -- because the two are never connected.

This module makes that one connection, by the one key both sides genuinely
share: the device entry symbol the hardware reports and the generated text
declares.  It refuses to guess.  A join that cannot be established for every
timed sample is reported as unresolved rather than approximated, because a
confident wrong attribution sends the whole search down the wrong branch.

Tasks that match no compiled symbol remain unattributed. They may include
framework operators, missing compiled sources or runtime helpers; a missing
symbol alone does not establish their ownership.
"""
from __future__ import annotations

from pathlib import Path
import os
import re

VERSION = "cross-level-attribution-v2"
NATIVE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".cxx"}
# A device entry point, as each backend's generated text declares it.
SYMBOL_PATTERNS = (re.compile(r"__global__\s+__aicore__\s+void\s+(\w+)\s*\("),
                   re.compile(r"__global__\s+void\s+(\w+)\s*\("))
UNEXPORTED = "unexported"


def enabled():
    return os.environ.get("KERNELBENCH_D1_ATTRIBUTION", "1") != "0"


def symbols(text):
    """Device entry symbols declared in one generated translation unit, in order."""
    return [name for name, _, _ in located_symbols(text)]


def signature(text, symbol):
    """The buffers a device entry takes, by the names the program gave them.

    This is the one thing in the generated code that points back up: the
    parameters carry the high-level buffer names, so `(Qh, Kh, Vh, O)` names
    the kernel in the author's own vocabulary where `main_kernel` -- which is
    what every kernel is called -- names nothing.
    """
    match = re.search(r"\b" + re.escape(symbol) + r"\s*\(([^)]*)\)", text or "")
    if not match:
        return ""
    names = []
    for part in match.group(1).split(","):
        word = part.strip().split()[-1] if part.strip() else ""
        word = word.lstrip("*").removesuffix("_handle")
        if word and word != "fftsAddr":
            names.append(word)
    return f"({', '.join(names)})" if names else ""


def located_symbols(text):
    """Each device entry as (symbol, first line, last line) of its definition.

    The line span is what turns "this kernel owns 82% of the forward" into
    somewhere to put the cursor, so it is read off the generated text rather
    than left to the reader to find.
    """
    text = text or ""
    found = []
    for pattern in SYMBOL_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), match.group(1)))
    located, starts = [], sorted(found)
    for index, (offset, name) in enumerate(starts):
        line = text[:offset].count("\n") + 1
        after = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        located.append((name, line, text[:after].count("\n") + 1))
    return located


def compiled_objects(checkpoint, export_record):
    """The editable source objects that own device work, in emission order.

    At High the objects are the exporter's kernels, each carrying the call site
    it came from.  At Native the agent edits the generated files directly, so
    the objects are those files.  Attribution therefore works on both sides of
    a transfer, which is what makes the two sides comparable at all.
    """
    objects = []
    if checkpoint.get("level") == "High":
        for index, kernel in enumerate(export_record.get("kernels", []) if export_record else []):
            text = kernel.get("device_text", "")
            for symbol in symbols(text):
                objects.append(dict(object_id=f"Native:kernel:{index}", symbol=symbol,
                                    high_call=f"High:call:{index}", callsite=kernel.get("callsite"),
                                    span=dict(kernel.get("native_span") or {},
                                              file=kernel.get("source")
                                              or (kernel.get("native_span") or {}).get("file")),
                                    signature=signature(text, symbol)))
        return objects
    for name, text in sorted((checkpoint.get("source_files") or {}).items()):
        if Path(name).suffix not in NATIVE_SUFFIXES:
            continue
        for symbol, line, end_line in located_symbols(text):
            objects.append(dict(object_id=f"Native:file:{name}:{symbol}", symbol=symbol,
                                high_call=None, callsite=None, signature=signature(text, symbol),
                                span=dict(file=name, line=line, end_line=end_line)))
    return objects


def _owned(rows, known):
    return [row for row in rows if row.get("kernel") in known]


def _join(samples, objects):
    """Per-sample task-to-object assignment, or a reason it cannot be made.

    The sequence of compiled tasks must be identical in every timed sample and
    every seed.  The evaluator already refuses a candidate whose launch
    sequence changes between trials, so a mismatch here means the compiled
    objects and the measured tasks are not describing the same program.
    """
    expected = [item["symbol"] for item in objects]
    known = set(expected)
    grade = "symbol" if len(known) == len(expected) else "ordinal"
    first = None
    for sample in samples:
        observed = [row["kernel"] for row in _owned(sample.get("rows", []), known)]
        if first is None:
            first = observed
        elif observed != first:
            return None, grade, (f"the launch sequence differs between samples: {first} "
                                 f"then {observed} in {sample.get('window_id')}")
    if not first:
        return None, grade, "no measured task matches a compiled device symbol"
    if grade == "ordinal":
        # Repeated symbols are resolvable only with explicit launch-order
        # provenance. File order in a compiler cache is not launch order.
        if not all(item.get("high_call") for item in objects) or first != expected:
            return None, grade, "duplicate symbols lack an exact compiled launch-order mapping"
    return known, grade, None


def attribute(samples, objects):
    """Assign every measured device task to a source object, or to nothing."""
    if not objects:
        return dict(version=VERSION, status="no_compiled_objects", objects=[],
                    reason="no device entry symbol was found in the candidate's compiled text")
    known, grade, reason = _join(samples, objects)
    if known is None:
        return dict(version=VERSION, status="unresolved", objects=[], reason=reason)

    totals = {item["object_id"]: 0.0 for item in objects}
    counts = {item["object_id"]: 0 for item in objects}
    launches = {item["object_id"]: set() for item in objects}
    residual, residual_names, measured = 0.0, {}, 0.0
    for sample in samples:
        rows = sample.get("rows", [])
        owned = _owned(rows, known)
        by_symbol = {item["symbol"]: item["object_id"] for item in objects}
        assignment = {id(row): (by_symbol[row["kernel"]] if grade == "symbol"
                               else objects[index]["object_id"])
                      for index, row in enumerate(owned)}
        for launch_index, row in enumerate(rows):
            value = row["value"]
            measured += value
            owner = assignment.get(id(row))
            if owner is None:
                residual += value
                residual_names[row["kernel"]] = residual_names.get(row["kernel"], 0.0) + value
                row["source_object_id"], row["mapping_grade"] = UNEXPORTED, "unmapped_symbol"
                continue
            totals[owner] += value
            counts[owner] += 1
            launches[owner].add(launch_index)
            row["source_object_id"], row["mapping_grade"] = owner, grade

    n = max(len(samples), 1)
    table = []
    for item in objects:
        key = item["object_id"]
        if key in {entry["object_id"] for entry in table}:
            continue
        table.append(dict(object_id=key, symbol=item["symbol"], grade=grade,
                          high_call=item["high_call"], callsite=item["callsite"], span=item["span"],
                          signature=item.get("signature", ""),
                          launch_indices=sorted(launches[key]),
                          mean_ms=totals[key] / n, tasks_per_sample=counts[key] / n,
                          share_of_device_work=totals[key] / measured if measured else None))
    table.sort(key=lambda entry: entry["mean_ms"], reverse=True)
    return dict(version=VERSION, status="attributed", grade=grade, samples=len(samples),
                basis="unique device symbol, or exact recorded High launch order for duplicate symbols",
                objects=table,
                unexported=dict(mean_ms=residual / n,
                                share_of_device_work=residual / measured if measured else None,
                                names=sorted(residual_names, key=residual_names.get, reverse=True),
                                interpretation="unmapped device symbols; framework ownership is not established"),
                mean_device_work_ms=measured / n)


def bottleneck(record):
    """The object that owns the most device work, when one is established."""
    if not record or record.get("status") != "attributed":
        return None
    top = record["objects"][0] if record["objects"] else None
    unexported = record.get("unexported") or {}
    if top and (unexported.get("mean_ms") or 0) > top["mean_ms"]:
        return dict(object_id=UNEXPORTED, mean_ms=unexported["mean_ms"],
                    share_of_device_work=unexported.get("share_of_device_work"),
                    detail=", ".join(unexported.get("names", [])[:3]))
    if not top:
        return None
    return dict(object_id=top["object_id"], mean_ms=top["mean_ms"],
                share_of_device_work=top["share_of_device_work"],
                detail=top.get("callsite") or (top.get("span") or {}).get("file"))


def attach(context, checkpoint, export_record):
    """Resolve the evidence record's reserved source-mapping fields in place.

    The row schema already carries ``source_object_id`` and ``mapping_grade``
    and deliberately leaves them null; this fills them and nothing else, so a
    disabled D1 is byte-identical to the record produced before it existed.
    """
    stages = context.get("evaluation_stage_evidence") or {}
    samples = stages.get("samples") or []
    if not enabled():
        context["attribution"] = dict(version=VERSION, status="disabled", objects=[],
                                      reason="KERNELBENCH_D1_ATTRIBUTION=0")
        return context["attribution"]
    if stages.get("status") != "measured" or not samples:
        context["attribution"] = dict(version=VERSION, status="unavailable", objects=[],
                                      reason="no bound full-suite device measurement for this checkpoint")
        return context["attribution"]
    record = attribute(samples, compiled_objects(checkpoint, export_record))
    context["attribution"] = record
    if record["status"] != "attributed":
        return record
    resolved = {row["evidence_id"]: row["source_object_id"]
                for sample in samples for row in sample.get("rows", [])
                if row.get("evidence_id") and row.get("source_object_id")}
    for item in context.get("evidence", []):
        owner = resolved.get(item.get("id"))
        if not owner:
            continue
        item["observation"]["object_id"] = owner
        item["source_mapping"] = (f"device entry symbol join ({record['grade']})"
                                  if owner != UNEXPORTED else
                                  "unmapped device symbol; ownership unknown")
    return record
