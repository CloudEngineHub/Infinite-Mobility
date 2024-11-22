import sys
import subprocess

from math import ceil

fac_name = sys.argv[1]
number = int(sys.argv[2])
max_process = 10
if len(sys.argv) > 3:
    max_process = int(sys.argv[3])

for i in range(ceil(number / max_process)):
    cmd = ""
    for j in range(max_process):
        if i*max_process+j >= number:
            break
        if cmd != "":
            cmd += " & "
        cmd += f"python -m infinigen_examples.generate_individual_assets --output_folder outputs/ -f {fac_name} -n 1 --start_seed {i*max_process+j}"
    cmd += " & wait"
    subprocess.call(cmd, shell=True)
        
# subprocess.call(
#     f"python -m infinigen_examples.generate_individual_assets --output_folder outputs/ -f OfficeChairFactory -n 1 --start_seed {}".split(" ")
# )