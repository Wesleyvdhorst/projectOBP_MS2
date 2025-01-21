import random
import pandas as pd

def generate_data_single_sheet(jobs_count, machines_count, seed=43):
    """Generates random data for job scheduling and outputs it to a single Excel sheet."""
    random.seed(seed)

    # Create job and machine indices
    jobs = list(range(jobs_count))
    machines = list(range(machines_count))

    # Generate random data
    release_times = {j: random.randint(0, 8) for j in jobs}
    due_dates = {j: random.randint(8, 20) for j in jobs}
    weights = {j: random.randint(1, 6) for j in jobs}
    processing_times = {(j, m): random.randint(2, 6) for j in jobs for m in machines}

    # Prepare data for a single DataFrame
    combined_data = []
    for job in jobs:
        for machine in machines:
            combined_data.append({
                'Job': job,
                'Machine': machine,
                'Release Time': release_times[job],
                'Due Date': due_dates[job],
                'Weight': weights[job],
                'Processing Time': processing_times[(job, machine)]
            })

    combined_df = pd.DataFrame(combined_data)

    # Write to Excel
    output_path = 'XXX'
    combined_df.to_excel(output_path, sheet_name='Job Scheduling Data', index=False)
    print(f"Data has been generated and saved to '{output_path}'.")

if __name__ == "__main__":
    # Input variables
    jobs_count = int(input("Enter the number of jobs: "))
    machines_count = int(input("Enter the number of machines: "))

    # Generate and save data
    generate_data_single_sheet(jobs_count, machines_count)