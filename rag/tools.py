"""Deterministic tools the assistant can call instead of doing arithmetic in the language model."""
import statistics

CAPABILITY_SCHEMA = {
    "name": "process_capability",
    "description": "Compute mean, sample standard deviation, Cp and Cpk for measurements against spec limits.",
    "args": {"values": "list of numbers (at least 2)", "lsl": "lower spec limit", "usl": "upper spec limit"},
}


def process_capability(values, lsl, usl):
    """Cp = (USL-LSL)/(6s); Cpk = min(USL-mean, mean-LSL)/(3s), s = sample standard deviation.

    Matches the NIST/SEMATECH handbook definitions (section 6.1.6).
    """
    vals = [float(v) for v in values]
    lsl, usl = float(lsl), float(usl)
    if len(vals) < 2:
        raise ValueError("need at least 2 measurements")
    if usl <= lsl:
        raise ValueError("usl must be greater than lsl")
    mean = statistics.fmean(vals)
    s = statistics.stdev(vals)
    if s == 0:
        raise ValueError("standard deviation is zero; capability is undefined")
    cp = (usl - lsl) / (6 * s)
    cpk = min(usl - mean, mean - lsl) / (3 * s)
    return {"n": len(vals), "mean": round(mean, 4), "std_dev": round(s, 4),
            "cp": round(cp, 3), "cpk": round(cpk, 3)}


TOOLS = {"process_capability": process_capability}
