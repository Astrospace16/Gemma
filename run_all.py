"""
run_all.py: Complete 1-Click SciSilentBench Runner & Metrics Calculator.
Automatically detects available Gemma models on your Google AI Studio API Key,
runs Condition A and Condition B, and prints the exact paper tables.
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
import tempfile
import subprocess
from pathlib import Path
import numpy as np

# =====================================================================
API_KEY = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))


def run_code_check(code_str: str, test_str: str) -> bool:
    """Runs Python code + assertions in a subprocess and returns True if exit code == 0."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(code_str + "\n\n" + test_str)
        tmp_path = tmp.name
    try:
        res = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=15
        )
        return res.returncode == 0
    except Exception:
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# 1. BENCHMARK TASKS (All 5 Bug Categories: TIME_SCALE, RAD_DEG, UNIT, JD_MJD, FRAME)
TASKS = [
    {
        "task_id": "LK-042",
        "repo": "lightkurve",
        "bug_type": "TIME_SCALE",
        "prompt": "Fix `compute_barycentric_phase` so input Astropy Time objects are phase-folded around `t0_tdb_jd` in Barycentric Dynamical Time (TDB) rather than UTC.",
        "mutated_code": (
            "import numpy as np\nfrom astropy.time import Time\n\n"
            "def compute_barycentric_phase(time_obj: Time, period_days: float, t0_tdb_jd: float) -> np.ndarray:\n"
            "    dt = time_obj.utc.jd - t0_tdb_jd\n"
            "    return (dt % period_days) / period_days\n"
        ),
        "visible_test": (
            "t = Time(['2020-01-01T00:00:00', '2020-01-02T00:00:00'], format='isot', scale='utc')\n"
            "phases = compute_barycentric_phase(t, period_days=2.0, t0_tdb_jd=t.tdb.jd[0])\n"
            "assert phases.shape == (2,) and np.all((phases >= 0.0) & (phases <= 1.0))\n"
        ),
        "hidden_oracle": (
            "t = Time(['2020-01-01T00:00:00', '2020-06-01T00:00:00'], format='isot', scale='utc')\n"
            "p = compute_barycentric_phase(t, period_days=1.5, t0_tdb_jd=2458849.5)\n"
            "expected = ((t.tdb.jd - 2458849.5) % 1.5) / 1.5\n"
            "assert np.allclose(p, expected, atol=1e-9)\n"
        )
    },
    {
        "task_id": "AP-017",
        "repo": "astropy",
        "bug_type": "RAD_DEG",
        "prompt": "Ensure `angular_sep_rad` accurately computes angular separation in radians given RA/Dec offsets and declination in degrees.",
        "mutated_code": (
            "import numpy as np\n\n"
            "def angular_sep_rad(dra_deg: float, ddec_deg: float, dec_deg: float) -> float:\n"
            "    dra = dra_deg * np.cos(np.deg2rad(dec_deg))\n"
            "    ddec = np.deg2rad(ddec_deg)\n"
            "    return float(np.hypot(dra, ddec))\n"
        ),
        "visible_test": (
            "assert angular_sep_rad(0.0, 0.0, 0.0) == 0.0\n"
            "assert angular_sep_rad(1.0, 1.0, 0.0) > 0.0\n"
        ),
        "hidden_oracle": (
            "res = angular_sep_rad(1.0, 0.0, 60.0)\n"
            "expected = np.deg2rad(1.0) * 0.5\n"
            "assert np.isclose(res, expected, rtol=1e-7)\n"
        )
    },
    {
        "task_id": "AP-051",
        "repo": "astropy",
        "bug_type": "UNIT",
        "prompt": "Fix `parallax_to_distance_pc` so that input Quantity angles (in arcsec or milliarcsec) are properly converted to arcseconds before computing distance in parsecs.",
        "mutated_code": (
            "import astropy.units as u\n\n"
            "def parallax_to_distance_pc(parallax: u.Quantity) -> u.Quantity:\n"
            "    p_arcsec = parallax.value\n"
            "    return (1.0 / p_arcsec) * u.pc\n"
        ),
        "visible_test": (
            "d = parallax_to_distance_pc(0.5 * u.arcsec)\n"
            "assert d.unit == u.pc and d.value == 2.0\n"
        ),
        "hidden_oracle": (
            "import numpy as np\n"
            "d_mas = parallax_to_distance_pc(10.0 * u.mas)\n"
            "assert np.isclose(d_mas.to_value(u.pc), 100.0, rtol=1e-7)\n"
        )
    },
    {
        "task_id": "LK-088",
        "repo": "lightkurve",
        "bug_type": "JD_MJD",
        "prompt": "Fix `time_to_kepler_bjd` so that the Astropy Time object returns Julian Date (JD) minus the Kepler offset 2454833.0, rather than Modified Julian Date (MJD).",
        "mutated_code": (
            "from astropy.time import Time\n\n"
            "def time_to_kepler_bjd(t: Time) -> float:\n"
            "    return float(t.tdb.mjd - 2454833.0)\n"
        ),
        "visible_test": (
            "t1 = Time('2010-01-01T00:00:00', scale='tdb')\n"
            "t2 = Time('2010-01-02T00:00:00', scale='tdb')\n"
            "assert round(time_to_kepler_bjd(t2) - time_to_kepler_bjd(t1), 5) == 1.0\n"
        ),
        "hidden_oracle": (
            "t = Time(2454833.0, format='jd', scale='tdb')\n"
            "assert abs(time_to_kepler_bjd(t)) < 1e-9\n"
        )
    },
    {
        "task_id": "SP-019",
        "repo": "SunPy",
        "bug_type": "FRAME",
        "prompt": "Fix `solar_synodic_rotation_deg` to use the synodic Carrington rotation rate (13.20 deg/day) instead of sidereal Stonyhurst rate (14.184 deg/day) when `frame='carrington'`.",
        "mutated_code": (
            "def solar_synodic_rotation_deg(dt_days: float, frame: str = 'carrington') -> float:\n"
            "    rate = 14.184 if frame == 'carrington' else 14.184\n"
            "    return dt_days * rate\n"
        ),
        "visible_test": (
            "assert solar_synodic_rotation_deg(0.0, 'carrington') == 0.0\n"
            "assert solar_synodic_rotation_deg(2.0, 'carrington') > 0.0\n"
        ),
        "hidden_oracle": (
            "assert abs(solar_synodic_rotation_deg(10.0, 'carrington') - 132.0) < 1e-6\n"
        )
    }
]


def discover_gemma_models() -> list:
    """Queries Google AI Studio to find the exact Gemma model IDs active on your API key."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            all_models = [
                m["name"].split("/")[-1]
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            gemma_models = [m for m in all_models if "gemma" in m.lower()]
            if gemma_models:
                return sorted(gemma_models)
            return all_models[:3]
    except Exception as e:
        print(f"Error querying model list: {e}")
        return ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]


def call_gemini_api(model: str, sys_prompt: str, user_prompt: str, temperature: float = 0.2):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": sys_prompt + "\n\n" + user_prompt}]}],
        "generationConfig": {"temperature": temperature}
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
        return text, tokens


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith("json"):
                p_strip = p_strip[4:].strip()
            if p_strip.startswith("{") and p_strip.endswith("}"):
                cleaned = p_strip
                break
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return {}
    return {}


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 5) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin])) * prop_in_bin
    return float(ece)


def compute_auroc(confidences: np.ndarray, labels: np.ndarray) -> float:
    pos = confidences[labels == 1]
    neg = confidences[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u_stat = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(u_stat / (len(pos) * len(neg)))


def run_task(task: dict, model: str, condition: str, temperature: float = 0.2) -> dict:
    t0 = time.time()
    total_tokens = 0

    if condition == "A":
        sys_p = (
            "You are a scientific coding agent. Fix or verify the Python function so it passes the visible unit test.\n"
            "Return ONLY a valid JSON object with keys:\n"
            '{"patched_code": "<full python code>", "confidence": 0.85}'
        )
        user_p = f"Task: {task['prompt']}\n\nCode:\n{task['mutated_code']}\n\nVisible Test:\n{task['visible_test']}"
        raw, tok = call_gemini_api(model, sys_p, user_p, temperature)
        total_tokens += tok
        parsed = extract_json(raw)
        patched_code = parsed.get("patched_code", task["mutated_code"])
        confidence = float(parsed.get("confidence", 0.75))
        decision = "SUBMIT"
    else:
        # Condition B: Invariant Synthesis + Self-Verification + Calibrated Abstention
        sys_p1 = (
            "You are a physics-aware coding agent. Step 1: Write a `patched_code` AND a `physics_invariant_test` "
            "(Python assertions checking dimensional unit scaling, non-zero angles, or time scale equivalence).\n"
            "Return ONLY a valid JSON object:\n"
            '{"patched_code": "<full python code>", "physics_invariant_test": "<python assertions>", "confidence": 0.90}'
        )
        user_p1 = f"Task: {task['prompt']}\n\nCode:\n{task['mutated_code']}\n\nVisible Test:\n{task['visible_test']}"
        raw1, tok1 = call_gemini_api(model, sys_p1, user_p1, temperature)
        total_tokens += tok1
        parsed1 = extract_json(raw1)
        patched_code = parsed1.get("patched_code", task["mutated_code"])
        inv_test = parsed1.get("physics_invariant_test", "assert False")
        confidence = float(parsed1.get("confidence", 0.80))

        inv_passed = run_code_check(patched_code, inv_test)
        vis_passed = run_code_check(patched_code, task["visible_test"])

        if not (inv_passed and vis_passed):
            sys_p2 = (
                "Your physics invariant or test failed! Either fix the physics bug in `patched_code` "
                "or set `decision` to 'ABSTAIN'. Return ONLY a valid JSON object:\n"
                '{"patched_code": "<full python code>", "decision": "SUBMIT", "confidence": 0.50}'
            )
            user_p2 = f"Previous code:\n{patched_code}\nInvariant:\n{inv_test}\nInvariant Passed: {inv_passed}"
            raw2, tok2 = call_gemini_api(model, sys_p2, user_p2, temperature)
            total_tokens += tok2
            parsed2 = extract_json(raw2)
            patched_code = parsed2.get("patched_code", patched_code)
            decision = str(parsed2.get("decision", "ABSTAIN")).upper()
            confidence = float(parsed2.get("confidence", 0.45))
        else:
            decision = "SUBMIT"

    dt = round(time.time() - t0, 2)
    vis_ok = run_code_check(patched_code, task["visible_test"])
    ora_ok = run_code_check(patched_code, task["hidden_oracle"])

    sub = (decision == "SUBMIT")
    return {
        "task_id": task["task_id"],
        "bug_type": task["bug_type"],
        "true_fix": sub and vis_ok and ora_ok,
        "silent_fail": sub and vis_ok and (not ora_ok),
        "abstained": decision == "ABSTAIN",
        "correct": vis_ok and ora_ok,
        "confidence": confidence,
        "tokens": total_tokens,
        "time_s": dt
    }


if __name__ == "__main__":
    if not API_KEY.strip():
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment.")
        sys.exit(1)

    models = discover_gemma_models()
    print(f"Detected Gemma Models on your API Key: {models}\n")

    header = (
        f"{'Model':<22} | {'Cond':<4} | {'TFR(%)':<6} | {'SFR(%)':<6} | "
        f"{'Abst(%)':<7} | {'AUROC':<5} | {'ECE':<5} | {'Tokens':<6} | {'Time(s)':<6}"
    )
    print(header)
    print("-" * len(header))

    all_runs = {}
    for m in models:
        for cond in ["A", "B"]:
            results = []
            for t in TASKS:
                try:
                    res = run_task(t, m, cond)
                    results.append(res)
                    time.sleep(1.5)  # Respect free-tier rate limits
                except urllib.error.HTTPError as e:
                    err_body = e.read().decode("utf-8", errors="ignore")
                    print(f"  [HTTP {e.code}] {m} on {t['task_id']}: {err_body[:120]}")
                except Exception as e:
                    print(f"  [Warn] {m} on {t['task_id']}: {e}")

            if results:
                n = len(results)
                tfr = 100.0 * sum(r["true_fix"] for r in results) / n
                sfr = 100.0 * sum(r["silent_fail"] for r in results) / n
                ab = 100.0 * sum(r["abstained"] for r in results) / n
                confs = np.array([r["confidence"] for r in results], dtype=float)
                labels = np.array([1 if r["correct"] else 0 for r in results], dtype=int)
                auroc = compute_auroc(confs, labels)
                ece = compute_ece(confs, labels)
                tok = np.mean([r["tokens"] for r in results])
                tm = np.mean([r["time_s"] for r in results])

                print(
                    f"{m:<22} | {cond:<4} | {tfr:<6.1f} | {sfr:<6.1f} | "
                    f"{ab:<7.1f} | {auroc:<5.2f} | {ece:<5.2f} | {tok:<6.0f} | {tm:<6.2f}"
                )
                all_runs[f"{m}_{cond}"] = results

    Path("experiment_results.json").write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    print("\nSaved full experiment details to experiment_results.json!")