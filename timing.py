import subprocess
import sys
import pandas as pd
import time
import os

#Settings Change for each Run
program = "LebwohlLasher.py"
output_file = "Results/Timings/Original.csv"
sizes = [25, 50, 100, 200]
steps = 50
temp = 0.5
repeats = 3

os.makedirs("Results/Timings", exist_ok=True)

rows =[]
for size in sizes:
    for r in range(1, repeats + 1):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, program, 
                                 str(steps),
                                 str(size), str(temp), "0"],
                                capture_output=True, text=True)
        wall = time.perf_counter() - start
        
        if result.returncode != 0:
            print(result.stderr)
            sys.exit("Run failed, see error above")
            
        out = result.stdout
        def grab(label):
            return float(out.split(label + ":")[1].split("s")[0])
        
        rows.append({"size": size, "repeat": r,
                     "wall_s": wall,
                     "setup_s": grab("setup"),
                     "loop_s": grab("Time"),
                     "mc_step_s": grab("MC_step"),
                     "all_energy_s": grab("all_energy"),
                     "get_order_s": grab("get_order"),
                     "savedat_s": grab("savedat")})
        print(f"Size {size}, repeat {r}: loop {grab('Time'):.3f} s, whole run {wall:.3f} s")
        
df = pd.DataFrame(rows)
df["other_s"] = df["wall_s"] - df["setup_s"] - df["loop_s"] - df["savedat_s"]
df.to_csv(output_file, index=False)
print("Saved", output_file)

#summary
print(df.groupby("size").mean().round(4))
         
        
