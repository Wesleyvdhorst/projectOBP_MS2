from ortools.sat.python import cp_model
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


class ORToolsScheduler:
    def __init__(self, num_jobs, num_machines):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.jobs_data = None
        self.weights = None
        self.due_dates = None
        self.release_times = None

    def generate_random_data(self, seed=42):
        random.seed(seed)

        # Generate processing times for each job on each machine
        self.jobs_data = []
        for _ in range(self.num_jobs):
            job_tasks = []
            for machine in range(self.num_machines):
                # Each entry is (machine_id, processing_time)
                job_tasks.append((machine, random.randint(2, 6)))
            self.jobs_data.append(job_tasks)

        # Generate weights, due dates, and release times
        self.weights = [random.randint(1, 6) for _ in range(self.num_jobs)]
        self.due_dates = [random.randint(8, 20) for _ in range(self.num_jobs)]
        self.release_times = [random.randint(0, 8) for _ in range(self.num_jobs)]

    def solve(self, time_limit_seconds=30):
        # Create the model
        model = cp_model.CpModel()

        # Define the horizon (upper bound for scheduling)
        horizon = sum(task[1] for job in self.jobs_data for task in job) * 2

        # Create job intervals and variables
        all_tasks = {}
        job_ends = []

        # Store start variables for solution extraction
        start_vars = {}

        for job_id in range(self.num_jobs):
            job_tasks = []
            for task_id, task in enumerate(self.jobs_data[job_id]):
                machine, duration = task

                # Create start variable and store it
                start = model.NewIntVar(0, horizon, f'start_{job_id}_{task_id}')
                start_vars[(job_id, task_id)] = start

                end = model.NewIntVar(0, horizon, f'end_{job_id}_{task_id}')
                interval = model.NewIntervalVar(start, duration, end, f'interval_{job_id}_{task_id}')

                all_tasks.setdefault(machine, []).append(interval)
                job_tasks.append((start, end, interval))

                if task_id == len(self.jobs_data[job_id]) - 1:
                    job_ends.append(end)

                # Add release time constraint for first task of each job
                if task_id == 0:
                    model.Add(start >= self.release_times[job_id])

                # Add precedence constraint
                if task_id > 0:
                    model.Add(start >= job_tasks[task_id - 1][1])

        # Add no-overlap constraints for machines
        for machine in range(self.num_machines):
            if machine in all_tasks:
                model.AddNoOverlap(all_tasks[machine])

        # Create tardiness variables
        tardiness = []
        for job_id in range(self.num_jobs):
            tard = model.NewIntVar(0, horizon, f'tardiness_{job_id}')
            model.Add(tard >= job_ends[job_id] - self.due_dates[job_id])
            tardiness.append(tard)

        # Objective: minimize weighted sum of tardiness
        objective_terms = []
        for job_id in range(self.num_jobs):
            objective_terms.append(tardiness[job_id] * self.weights[job_id])
        model.Minimize(sum(objective_terms))

        # Create a solver and solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        status = solver.Solve(model)

        # Process the solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            schedule = []
            for job_id in range(self.num_jobs):
                for task_id, task in enumerate(self.jobs_data[job_id]):
                    machine, duration = task
                    start_time = solver.Value(start_vars[(job_id, task_id)])

                    schedule.append({
                        'Job': job_id,
                        'Machine': machine,
                        'Start Time': start_time,
                        'Processing Time': duration,
                        'End Time': start_time + duration,
                        'Release Time': self.release_times[job_id],
                        'Due Date': self.due_dates[job_id],
                        'Weight': self.weights[job_id]
                    })

            return pd.DataFrame(schedule), solver.ObjectiveValue()
        else:
            raise ValueError("No solution found!")

    def visualize_schedule(self, schedule_df):
        fig, ax = plt.subplots(figsize=(15, 8))

        # Define colors for jobs
        unique_jobs = schedule_df['Job'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_jobs)))

        # Plot jobs
        for index, row in schedule_df.iterrows():
            job_color = colors[int(row['Job'])]
            ax.barh(y=row['Machine'],
                    width=row['Processing Time'],
                    left=row['Start Time'],
                    height=0.3,
                    color=job_color,
                    edgecolor='black',
                    label=f'Job {int(row["Job"])}' if row['Machine'] == 0 else "")

            # Add job labels
            ax.text(row['Start Time'] + row['Processing Time'] / 2,
                    row['Machine'],
                    f'J{int(row["Job"])}',
                    ha='center',
                    va='center',
                    color='black',
                    fontweight='bold')

        # Customize chart
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('OR-Tools Job Schedule')
        ax.set_yticks(range(self.num_machines))
        ax.set_yticklabels([f'Machine {i}' for i in range(self.num_machines)])
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  title="Jobs",
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')

        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create scheduler
    scheduler = ORToolsScheduler(num_jobs=20, num_machines=3)

    # Generate random data
    scheduler.generate_random_data()

    # Solve and time the execution
    start_time = time.time()
    schedule_df, objective_value = scheduler.solve(time_limit_seconds=30)
    solve_time = time.time() - start_time

    # Print results
    print(f"\nSolution found in {solve_time:.2f} seconds")
    print(f"Objective value (weighted tardiness): {objective_value}")

    # Visualize the schedule
    scheduler.visualize_schedule(schedule_df)
    plt.show()