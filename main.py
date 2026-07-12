#! python3

"""
This is the master script file for the project.

Wang Wenzhi
Time: 2026-07-12
"""

import os
from pathlib import Path
import getpass

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Set up the current working directory
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-1-1. Set up the project root
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

user_name = getpass.getuser()

if user_name == "wang":
    PROJECT_ROOT = Path("E:/Dropbox/E_Projects/TalentDiscovery")
else:
    PROJECT_ROOT = Path.cwd()

os.chdir(PROJECT_ROOT)
print(f"Current working directory: {Path.cwd().as_posix()}.")

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-1-2. Define functions to deal with paths
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


def relative_path(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Define paths-related variables
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. Codes
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

DIR_CODES = PROJECT_ROOT / "codes"

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Outputs
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

DIR_OUTPUTS = PROJECT_ROOT / "outputs"

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-3. Data
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

DIR_DATA = PROJECT_ROOT / "data"
DIR_RAWDATA = DIR_DATA / "a_raw_data"
DIR_TEMPDATA = DIR_DATA / "b_temp_data"
DIR_FINALDATA = DIR_DATA / "c_final_data"
