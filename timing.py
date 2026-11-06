import subprocess
import sys
import pandas as pd
import time
import os
import platform

#File names
program = sys.argv[1]
label = sys.argv[2]
threads = int(sys.argv[3]) if len(sys.argv) > 3 else None #for parallel version
output_file = "Results/Timings/{}.csv".format(label)

#Settings keep same so all same
sizes = [400, 800, 1600]
steps = 50
temp = 0.5
repeats = 3

os.makedirs("Results/Timings", exist_ok=True)

rows =[]
for size in sizes:
    for r in range(1, repeats + 1):
        command = [sys.executable, program, str(steps), str(size), str(temp), "0"]
        if threads is not None:
            command.append(str(threads))
        start = time.perf_counter()
        result = subprocess.run(command,capture_output=True, text=True)
                                
        wall = time.perf_counter() - start
        
        if result.returncode != 0:
            print(result.stderr)
            sys.exit("Run failed, see error above")
            
        out = result.stdout
        
        if "Time:" not in out: #time saftety check
            print(out)
            sys.exit("Program ran but no printed timings, check command line args")
        
        def grab(name):
            if name + ":" not in out:
                return 0.0
            return float(out.split(name + ":")[1].split("s")[0])
        
        rows.append({"version": label,
                     "size": size, "repeat": r,
                     "threads": threads if threads is not None else 1,
                     "steps": steps, "temp": temp,
                     "wall_s": wall,
                     "compile_s": grab("compile"),
                     "setup_s": grab("setup"),
                     "loop_s": grab("Time"),
                     "mc_step_s": grab("MC_step"),
                     "all_energy_s": grab("all_energy"),
                     "get_order_s": grab("get_order"),
                     "savedat_s": grab("savedat")})
        print(f"Size {size}, repeat {r}: loop {grab('Time'):.3f} s, whole run {wall:.3f} s")
        
df = pd.DataFrame(rows)
df["other_s"] = df["wall_s"] -  df["compile_s"] - df["setup_s"] - df["loop_s"] - df["savedat_s"]
df.to_csv(output_file, index=False)
print("Saved", output_file)

import numpy
with open("Results/Timings/machine_info.txt", "w") as f:
    f.write("CPU: {}\n".format(platform.processor()))
    f.write("Cores: {}\n".format(os.cpu_count()))
    try:
        import psutil
        f.write("Physical cores: {}\n".format(psutil.cpu_count(logical=False)))
    except ImportError:
        pass
    f.write("OS: {}\n".format(platform.platform()))
    f.write("Python: {}\n".format(platform.python_version()))
    f.write("NumPy: {}\n".format(numpy.__version__))
    try: 
        import numba
        f.write("Numba: {}\n".format(numba.__version__))
    except ImportError:
        pass

#summary
print(df.groupby("size")[["compile_s", "loop_s", "mc_step_s", "all_energy_s", "get_order_s", "wall_s"]].mean().round(4))
         
        
