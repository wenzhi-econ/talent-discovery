import subprocess
import sys
from pathlib import Path

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Specify paths and parameters in the command
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>

TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

CODE_DIR = main.DIR_CODES / "A03_Industry_Automation"

PIPELINE_SCRIPTS = [
    "B01_CompanyCountsByCountry.py",
    "B02_CompaniesWithEmployeesByCountry.py",
    "B03_CompaniesWithPostingsByCountry.py",
    "B04_ActiveCompaniesByCountry.py",
    "C01_EmployeeDistributionAllCompanies.py",
    "C03_FocalEmployeeDistributionAllCompanies.py",
    "D01_IndeedPostingsMonthly.py",
    "D02_IndeedFocalPostingsMonthly.py",
    "E01_LinkedInPostingsMonthly.py",
    "E02_LinkedInFocalPostingsMonthly.py",
]

# >> S-1-2. The virtual environment

CONDA_ENV = "Talent"

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. A function to run a script file with the specified parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def run_pipeline_script(script_name: str) -> None:
    """Run one pipeline script in the CompAdvantages conda environment."""
    script_path = CODE_DIR / script_name
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        str(main.relative_path(script_path)),
    ]

    print("=" * 72)
    print(f"Running {script_name}")
    print("=" * 72)

    completed = subprocess.run(command, cwd=main.PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"{script_name} failed with exit code {completed.returncode}.")

    print()


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Run scripts in this folder to clean data
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>

for script_name in PIPELINE_SCRIPTS:
    run_pipeline_script(script_name)

print("Data-visualization pipeline completed successfully.")
