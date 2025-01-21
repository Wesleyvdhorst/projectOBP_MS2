import pulp
import random
import time
import matplotlib.pyplot as plt

# Function to generate random data for the job scheduling problem
def generate_data(num_jobs, num_machines):
    jobs = list(range(num_jobs))
    machines = list(range(num_machines))

    release_times = {j: random.randint(0, 8) for j in jobs}
    due_dates = {j: random.randint(8, 20) for j in jobs}
    weights = {j: random.randint(1, 6) for j in jobs}
    processing_times = {(j, m): random.randint(2, 6) for j in jobs for m in machines}

    return jobs, machines, release_times, due_dates, weights, processing_times

# Function to solve the job scheduling problem and return the runtime
def solve_job_scheduling(num_jobs, num_machines):
    jobs, machines, release_times, due_dates, weights, processing_times = generate_data(num_jobs, num_machines)

    # Define ILP Model
    model = pulp.LpProblem("Job_Scheduling", pulp.LpMinimize)

    # Decision Variables
    S = pulp.LpVariable.dicts("Start", [(j, m) for j in jobs for m in machines], lowBound=0, cat='Continuous')
    C = pulp.LpVariable.dicts("Completion", jobs, lowBound=0, cat='Continuous')
    T = pulp.LpVariable.dicts("Tardiness", jobs, lowBound=0, cat='Continuous')
    X = pulp.LpVariable.dicts("Order", [(j, j2, m) for j in jobs for j2 in jobs for m in machines if j != j2], cat='Binary')

    # Objective Function: Minimize Weighted Sum of Tardiness
    model += pulp.lpSum(weights[j] * T[j] for j in jobs)

    # Constraints
    BIG_M = 1000
    for j in jobs:
        model += C[j] == S[j, num_machines-1] + processing_times[j, num_machines-1]
        model += T[j] >= C[j] - due_dates[j]

        for m in machines:
            if m == 0:
                model += S[j, m] >= release_times[j]
            else:
                model += S[j, m] >= S[j, m-1] + processing_times[j, m-1]

            for j2 in jobs:
                if j != j2:
                    model += S[j, m] + processing_times[j, m] <= S[j2, m] + BIG_M * (1 - X[j, j2, m])
                    model += S[j2, m] + processing_times[j2, m] <= S[j, m] + BIG_M * X[j, j2, m]

    # Start timing the solver
    start_time = time.time()

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    solve_time = time.time() - start_time  # Time taken to solve the model
    return solve_time

# Main function to test for different configurations and compare runtimes
def test_runtime():
    job_server_combinations = [
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
        (8, 3),
        (9, 3),
    ]
    
    runtimes = []

    for num_jobs, num_machines in job_server_combinations:
        print(f"Testing with {num_jobs} jobs and {num_machines} machines...")
        solve_time = solve_job_scheduling(num_jobs, num_machines)
        runtimes.append((num_jobs, num_machines, solve_time))
        print(f"Runtime: {solve_time:.2f} seconds\n")
    
    # Plotting the runtime comparison
    job_counts = [x[0] for x in runtimes]
    machine_counts = [x[1] for x in runtimes]
    times = [x[2] for x in runtimes]

    plt.figure(figsize=(10, 6))
    plt.scatter(job_counts, times, c=machine_counts, cmap='viridis', label="Machines")
    plt.title("Runtime vs. Number of Jobs")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Runtime (seconds)")
    plt.colorbar(label="Number of Machines")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_runtime()
