import pulp
import random
import pandas as pd

# --------------------------
# 1. Generate Fake Data
# --------------------------
random.seed(42)

J = 5  # Number of jobs
M = 3  # Number of machines

jobs = list(range(J))
machines = list(range(M))

# Generate random release times, due dates, weights, and processing times
release_times = {j: random.randint(0, 5) for j in jobs}
due_dates = {j: random.randint(10, 20) for j in jobs}
weights = {j: random.randint(1, 5) for j in jobs}
processing_times = {(j, m): random.randint(2, 6) for j in jobs for m in machines}

# Large M constant
BIG_M = 1000

# --------------------------
# 2. Define the ILP Model
# --------------------------
model = pulp.LpProblem("Job_Scheduling", pulp.LpMinimize)

# Decision Variables
S = pulp.LpVariable.dicts("Start", [(j, m) for j in jobs for m in machines], lowBound=0, cat='Continuous')
C = pulp.LpVariable.dicts("Completion", jobs, lowBound=0, cat='Continuous')
T = pulp.LpVariable.dicts("Tardiness", jobs, lowBound=0, cat='Continuous')
X = pulp.LpVariable.dicts("Order", [(j, j2, m) for j in jobs for j2 in jobs for m in machines if j != j2], cat='Binary')

# Objective Function: Minimize Weighted Sum of Tardiness
model += pulp.lpSum(weights[j] * T[j] for j in jobs)

# Constraints
for j in jobs:
    # Completion time constraint
    model += C[j] == S[j, M-1] + processing_times[j, M-1]
    # Tardiness definition
    model += T[j] >= C[j] - due_dates[j]

    for m in machines:
        # Release time constraint (first machine only)
        if m == 0:
            model += S[j, m] >= release_times[j]
        else:
            # Enforce job sequence across machines
            model += S[j, m] >= S[j, m-1] + processing_times[j, m-1]

        for j2 in jobs:
            if j != j2:
                # Machine order constraints
                model += S[j2, m] >= S[j, m] + processing_times[j, m] - BIG_M * (1 - X[j, j2, m])

# --------------------------
# 3. Solve the Model
# --------------------------
solver = pulp.PULP_CBC_CMD(msg=False)
model.solve(solver)

# --------------------------
# 4. Print Results
# --------------------------
print("\nOptimal Schedule:")
for j in jobs:
    for m in machines:
        print(f"Job {j} starts on Machine {m} at {pulp.value(S[j, m]):.2f}")
    print(f"Completion: {pulp.value(C[j]):.2f}, Tardiness: {pulp.value(T[j]):.2f}\n")

# --------------------------
# 5. Convert to DataFrame for Visualization
# --------------------------
schedule_df = pd.DataFrame([{  
    "Job": j,   
    "Machine": m,   
    "Start Time": pulp.value(S[j, m]),   
    "Processing Time": processing_times[j, m],   
    "End Time": pulp.value(S[j, m]) + processing_times[j, m]  
} for j in jobs for m in machines])

print(schedule_df)
