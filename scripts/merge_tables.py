import pandas as pd
import numpy as np


path = "/data/lkolmar/datasets/realistic_topspin/"

num_threads = 9
index_offset = 0 # does the first thread start at 0 or 1?

if __name__ == "__main__":


    dfs = []
    for i in range(num_threads):
        df = pd.read_csv(path + f"config/simulation_pid{i+index_offset}.csv")
        dfs.append(df)

    print(f"Loaded {len(dfs)} dataframes with {len(dfs[0])} entries each.")
    merged_df = dfs[0].copy()
    for i in range(len(dfs[0])):
        data = []
        for j in range(num_threads):
            if dfs[j].iloc[i]["finished"] and not (dfs[j].iloc[i]["path"] == "not set"):
                # print(f"Simulation {i} finished in thread {j} with path {dfs[j].iloc[i]['path']}")
                # break
                for d in data:
                    if d["finished"]:
                        # print(f"Warning: Simulation {i} has duplicated entries")
                        pass
                data.append(dfs[j].iloc[i])
                merged_df.iloc[i] = dfs[j].iloc[i]
        
        if not merged_df.iloc[i]["finished"] or merged_df.iloc[i]["path"] == "not set":
            print(f"Warning: Simulation {i} is not finished in any thread")
            merged_df.loc[i, "finished"] = False

    merged_df.to_csv(path + "config/simulation.csv", index=False)
 