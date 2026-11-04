import subprocess
import sys
import pandas as pd
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
        result = subprocess.run([sys.executable, program, 
                                 str(steps),
                                 str(size), str(temp), "0"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            sys.exit("Run failed, see error above")
            
        time_taken = float(result.stdout.split("Time:")[1].split("s")[0])
        rows.append({"size": size, "repeat": r, "time_s": time_taken})
        print(f"Size {size}, repeat {r}: {time_taken:.3f} s")
        
df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)
print("Saved", output_file)

#summary
print(df.groupby("size")["time_s"].agg(["mean", "std", "min", "max"]))
         
        
