"""
mutator.py: AST-level physical bug injector and Silence Condition verifier.
"""
import ast
import subprocess
import sys
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Callable, Optional


class PhysicsBugMutator(ast.NodeTransformer):
    """Injects domain-specific silent physical mutations into Python ASTs."""

    def __init__(self, bug_type: str):
        self.bug_type = bug_type
        self.mutated = False

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # 1. TIME_SCALE: Conflate Barycentric Dynamical Time (tdb) with UTC (utc)
        if self.bug_type == "TIME_SCALE" and node.attr == "tdb":
            node.attr = "utc"
            self.mutated = True

        # 2. JD_MJD: Swap Julian Date (.jd) with Modified Julian Date (.mjd)
        elif self.bug_type == "JD_MJD" and node.attr == "jd":
            node.attr = "mjd"
            self.mutated = True

        # 3. FRAME: Swap ICRS and GCRS or Carrington and Stonyhurst
        elif self.bug_type == "FRAME" and node.attr == "icrs":
            node.attr = "gcrs"
            self.mutated = True

        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.mutated:
            return node

        # 4. RAD_DEG: Strip np.deg2rad or np.rad2deg conversions
        if self.bug_type == "RAD_DEG":
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("deg2rad", "rad2deg"):
                if node.args:
                    self.mutated = True
                    return node.args[0]  # Return un-converted angle

        # 5. UNIT: Replace .to(u.unit) with .to_value() stripping units or scaling
        if self.bug_type == "UNIT":
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                self.mutated = True
                return node.func.value

        return node


def run_code_check(code_str: str, test_str: str) -> bool:
    """Executes code + test string in an isolated subprocess and returns True if exit code == 0."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
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
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def generate_silent_bug_task(
    task_id: str,
    repo: str,
    bug_type: str,
    prompt: str,
    clean_code: str,
    visible_test: str,
    hidden_oracle: str,
    target_file: str = ""
) -> Optional[Dict]:
    """
    Applies a physics mutation and verifies the Silence Condition:
    Pass(Visible Test) == True AND Pass(Hidden Oracle) == False.
    """
    tree = ast.parse(clean_code)
    mutator = PhysicsBugMutator(bug_type=bug_type)
    mutated_tree = mutator.visit(tree)
    ast.fix_missing_locations(mutated_tree)

    if not mutator.mutated:
        print(f"[{task_id}] Mutation {bug_type} did not trigger on target AST.")
        return None

    mutated_code = ast.unparse(mutated_tree)

    # Verify clean code passes both
    assert run_code_check(clean_code, visible_test), f"[{task_id}] Clean code failed visible test!"
    assert run_code_check(clean_code, hidden_oracle), f"[{task_id}] Clean code failed oracle!"

    # Check Silence Condition on mutated code
    vis_passes = run_code_check(mutated_code, visible_test)
    oracle_passes = run_code_check(mutated_code, hidden_oracle)

    if vis_passes and not oracle_passes:
        print(f"[{task_id}] VALID SILENT BUG GENERATED ({bug_type}): Visible=PASS, Oracle=FAIL")
        return {
            "task_id": task_id,
            "repo": repo,
            "target_file": target_file,
            "bug_type": bug_type,
            "prompt": prompt,
            "mutated_code": mutated_code,
            "clean_code": clean_code,
            "visible_test": visible_test,
            "hidden_oracle": hidden_oracle
        }
    else:
        print(f"[{task_id}] Rejected: Visible={vis_passes}, Oracle={oracle_passes}")
        return None


if __name__ == "__main__":
    # Expand the original two validated seeds with 23 additional scientific
    # unit/time conversion tasks. The code snippets are self-contained; their
    # target_file values are benchmark metadata, not external repository files.
    extra_specs = [
        ("ANG-001", "orbital_inclination_rad", "inclination_deg", 30.0, "Convert an orbital inclination from degrees to radians."),
        ("ANG-002", "detector_roll_rad", "roll_deg", 42.5, "Convert an instrument detector roll angle from degrees to radians."),
        ("ANG-003", "polarization_axis_rad", "axis_deg", 67.0, "Convert a polarization analyzer axis from degrees to radians."),
        ("ANG-004", "seismic_azimuth_rad", "azimuth_deg", 123.0, "Convert a seismic source azimuth from degrees to radians."),
        ("ANG-005", "wind_direction_rad", "direction_deg", 215.0, "Convert a meteorological wind direction from degrees to radians."),
        ("ANG-006", "spherical_latitude_rad", "latitude_deg", 51.25, "Convert a geodetic latitude from degrees to radians."),
        ("ANG-007", "beam_pointing_rad", "pointing_deg", 2.75, "Convert a telescope beam pointing offset from degrees to radians."),
        ("ANG-008", "pendulum_initial_angle_rad", "initial_angle_deg", 18.0, "Convert a pendulum's initial displacement angle from degrees to radians."),
        ("ANG-009", "gyroscope_yaw_rad", "yaw_deg", 271.0, "Convert a gyroscope yaw measurement from degrees to radians."),
        ("ANG-010", "magnetic_azimuth_rad", "azimuth_deg", 143.5, "Convert a magnetic field azimuth from degrees to radians."),
        ("JDM-001", "observation_julian_date", "time_obj", 2459000.25, "Return an observation timestamp as a Julian Date, preserving the JD epoch."),
        ("JDM-002", "transit_epoch_jd", "time_obj", 2459123.75, "Return an exoplanet transit epoch in Julian Date units."),
        ("JDM-003", "exposure_midpoint_jd", "time_obj", 2459345.125, "Return an exposure midpoint as a Julian Date."),
        ("JDM-004", "simulation_time_jd", "time_obj", 2460001.5, "Return a simulation timestamp in Julian Date rather than Modified Julian Date."),
        ("UNT-001", "mass_in_kg", "quantity", 1000.0, "Convert a mass quantity to kilograms."),
        ("UNT-002", "distance_in_m", "quantity", 2.5, "Convert a distance quantity to meters."),
        ("UNT-003", "energy_in_joule", "quantity", 3.0, "Convert an energy quantity to joules."),
        ("UNT-004", "pressure_in_pascal", "quantity", 1.0, "Convert a pressure quantity to pascals."),
        ("FRM-001", "icrs_ra_from_gcrs_1", "coordinate", "10,20,2024-01-01,9.999889832459793", "Convert a GCRS sky coordinate to ICRS right ascension."),
        ("FRM-002", "icrs_ra_from_gcrs_2", "coordinate", "45,-30,2024-04-01,45.00504029668637", "Transform a geocentric apparent direction to ICRS right ascension."),
        ("FRM-003", "icrs_ra_from_gcrs_3", "coordinate", "120,60,2024-07-01,120.01039590388469", "Report ICRS right ascension for a GCRS coordinate at the observation epoch."),
        ("FRM-004", "icrs_ra_from_gcrs_4", "coordinate", "270,10,2024-10-01,270.00089151751297", "Convert an observed GCRS direction into the ICRS frame."),
        ("FRM-005", "icrs_ra_from_gcrs_5", "coordinate", "188,2,2025-01-01,187.99983159389515", "Return the ICRS right ascension of a GCRS coordinate."),
    ]
    unit_cases = {
        "UNT-001": ("u.g", "u.kg", 1000.0, 1.0),
        "UNT-002": ("u.km", "u.m", 2.5, 2500.0),
        "UNT-003": ("u.erg", "u.J", 3.0e7, 3.0),
        "UNT-004": ("u.bar", "u.Pa", 1.0, 1.0e5),
    }

    seed_specs = []
    for task_id, function_name, parameter, value, prompt in extra_specs:
        if task_id.startswith("ANG-"):
            clean_code = (
                "import numpy as np\n\n"
                f"def {function_name}({parameter}: float) -> float:\n"
                f"    return float(np.deg2rad({parameter}))\n"
            )
            visible_test = f"assert {function_name}({value}) > 0.0\n"
            hidden_oracle = (
                f"assert np.isclose({function_name}({value}), np.deg2rad({value}), rtol=1e-12), "
                f"'degree/radian conversion incorrect'\n"
            )
            bug_type = "RAD_DEG"
        elif task_id.startswith("JDM-"):
            clean_code = (
                "from astropy.time import Time\n\n"
                f"def {function_name}({parameter}: Time) -> float:\n"
                f"    return float({parameter}.jd)\n"
            )
            visible_test = (
                f"t = Time({value}, format='jd', scale='tdb')\n"
                f"assert {function_name}(t) > 0.0\n"
            )
            hidden_oracle = (
                f"t = Time({value}, format='jd', scale='tdb')\n"
                f"assert abs({function_name}(t) - {value}) < 1e-8, 'JD/MJD epoch mismatch'\n"
            )
            bug_type = "JD_MJD"
        elif task_id.startswith("FRM-"):
            ra_text, dec_text, date_text, expected_text = value.split(",")
            ra, dec, expected = float(ra_text), float(dec_text), float(expected_text)
            clean_code = (
                "import astropy.units as u\n"
                "from astropy.coordinates import SkyCoord, GCRS\n"
                "from astropy.time import Time\n\n"
                f"def {function_name}(ra_deg: float, dec_deg: float, obstime: Time) -> float:\n"
                "    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=GCRS(obstime=obstime))\n"
                "    return float(coord.icrs.ra.deg)\n"
            )
            visible_test = (
                f"t = Time('{date_text}')\n"
                f"assert {function_name}({ra}, {dec}, t) > 0.0\n"
            )
            hidden_oracle = (
                f"t = Time('{date_text}')\n"
                f"assert abs({function_name}({ra}, {dec}, t) - {expected}) < 1e-8, 'ICRS/GCRS frame conversion incorrect'\n"
            )
            bug_type = "FRAME"
        else:
            input_unit, output_unit, magnitude, expected = unit_cases[task_id]
            clean_code = (
                "import astropy.units as u\n\n"
                f"def {function_name}({parameter}) -> float:\n"
                f"    return float({parameter}.to({output_unit}).value)\n"
            )
            visible_test = (
                f"q = {magnitude} * {input_unit}\n"
                f"assert {function_name}(q) > 0.0\n"
            )
            hidden_oracle = (
                f"q = {magnitude} * {input_unit}\n"
                f"assert abs({function_name}(q) - {expected}) <= max(1e-9, abs({expected}) * 1e-9), "
                f"'unit conversion incorrect'\n"
            )
            bug_type = "UNIT"

        seed_specs.append({
            "task_id": task_id,
            "repo": "SciSilentBench synthetic scientific module",
            "target_file": f"scientific_examples/{function_name}.py",
            "bug_type": bug_type,
            "prompt": prompt,
            "clean_code": clean_code,
            "visible_test": visible_test,
            "hidden_oracle": hidden_oracle,
        })

    with open("sample_tasks.json", "r", encoding="utf-8") as f:
        validated_tasks = json.load(f)

    # Revalidate the original tasks before carrying them into the larger set.
    for task in validated_tasks:
        assert run_code_check(task["clean_code"], task["visible_test"])
        assert run_code_check(task["clean_code"], task["hidden_oracle"])
        assert run_code_check(task["mutated_code"], task["visible_test"])
        assert not run_code_check(task["mutated_code"], task["hidden_oracle"])

    for spec in seed_specs:
        task = generate_silent_bug_task(**spec)
        if task:
            validated_tasks.append(task)

    if len(validated_tasks) < 25:
        raise RuntimeError(f"Only {len(validated_tasks)} validated tasks; need at least 25")

    with open("tasks_25.json", "w", encoding="utf-8") as f:
        json.dump(validated_tasks, f, indent=2)
    print(f"Saved {len(validated_tasks)} validated tasks to tasks_25.json")