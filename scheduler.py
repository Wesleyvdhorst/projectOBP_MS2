import pandas as pd
from ortools.sat.python import cp_model
import random
import matplotlib.pyplot as plt
import numpy as np
import time


class ORToolsScheduler:
    def __init__(self, jobs_data, weights, due_dates, release_times, num_jobs, num_machines):
        self.jobs_data = jobs_data
        self.weights = weights
        self.due_dates = due_dates
        self.release_times = release_times
        self.num_jobs = num_jobs
        self.num_machines = num_machines

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

        # Set x-axis to start at 0
        ax.set_xlim(left=0)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                title="Jobs",
                bbox_to_anchor=(1.05, 1),
                loc='upper left')

        plt.tight_layout()
        return fig

# Function to load the data from the Excel file
def load_data_from_excel(excel_file):
    # Read the Excel file
    df = pd.read_excel(excel_file, sheet_name='Job Scheduling Data')

    # Extract number of jobs and machines from the data
    num_jobs = df['Job'].nunique()
    num_machines = df['Machine'].nunique()

    # Initialize job data structure
    jobs_data = []
    for job_id in range(num_jobs):
        job_tasks = []
        for machine in range(num_machines):
            # Get processing time for the job and machine
            row = df[(df['Job'] == job_id) & (df['Machine'] == machine)]
            processing_time = row['Processing Time'].values[0]
            job_tasks.append((machine, processing_time))
        jobs_data.append(job_tasks)

    # Extract job information (weights, due dates, release times)
    weights = df.groupby('Job')['Weight'].first().tolist()
    due_dates = df.groupby('Job')['Due Date'].first().tolist()
    release_times = df.groupby('Job')['Release Time'].first().tolist()

    return jobs_data, weights, due_dates, release_times, num_jobs, num_machines


# Example usage for debugging without UI (File upload)
if __name__ == "__main__":
    # Replace with your own Excel file path
    excel_file = 'excel.xlsx'  # Change to your file path

    # Load data from Excel file
    try:
        jobs_data, weights, due_dates, release_times, num_jobs, num_machines = load_data_from_excel(excel_file)
    except Exception as e:
        print(f"Error loading data from Excel: {e}")
        exit()

    # Create the scheduler with the loaded data
    scheduler = ORToolsScheduler(jobs_data, weights, due_dates, release_times, num_jobs, num_machines)

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
