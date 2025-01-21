import random
import io
import base64
from flask import Flask, render_template, request, redirect, url_for
from scheduler import ORToolsScheduler  # Assuming the class exists
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

REQUIRED_COLUMNS = ['Job', 'Machine', 'Processing Time', 'Release Time', 'Due Date', 'Weight']

@app.route('/')
def landing_page():
    """Landing page to choose between file upload or generated data."""
    return render_template('landing_page.html')


@app.route('/upload_or_generate', methods=['POST'])
def upload_or_generate():
    """Handles the choice of file upload or data generation."""
    choice = request.form.get('choice')
    if choice == 'upload':
        return redirect(url_for('upload_page'))
    elif choice == 'generate':
        return redirect(url_for('generate_data'))
    return render_template('landing_page.html', error="Invalid selection.")


@app.route('/generate_data', methods=['GET', 'POST'])
def generate_data():
    """Generates random data or displays the random data form."""
    if request.method == 'GET':
        return render_template('generate_data.html')

    try:
        num_jobs = int(request.form['num_jobs'])
        num_machines = int(request.form['num_machines'])

        # Validate input
        if num_jobs <= 0 or num_machines <= 0:
            raise ValueError("Number of jobs and machines must be positive integers.")

        # Generate random data
        random.seed(43)
        jobs = list(range(num_jobs))
        machines = list(range(num_machines))
        release_times = {j: random.randint(0, 8) for j in jobs}
        due_dates = {j: random.randint(8, 20) for j in jobs}
        weights = {j: random.randint(1, 6) for j in jobs}
        processing_times = {(j, m): random.randint(2, 6) for j in jobs for m in machines}

        data = []
        for j in jobs:
            for m in machines:
                data.append({
                    'Job': j,
                    'Machine': m,
                    'Processing Time': processing_times[(j, m)],
                    'Release Time': release_times[j],
                    'Due Date': due_dates[j],
                    'Weight': weights[j]
                })

        df = pd.DataFrame(data)
        
        # Convert the dataframe to JSON and ensure it's properly passed as part of the URL
        df_json = df.to_json()
        return redirect(url_for('loading_screen', dataframe=df_json))

    except ValueError as e:
        return render_template('generate_data.html', error=str(e))


@app.route('/upload_page', methods=['GET'])
def upload_page():
    """Displays the upload page to select an Excel file."""
    return render_template('upload_page.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and redirects to the loading page."""
    file, error = validate_file(request)
    if error:
        return render_template('upload_page.html', error=error)

    df, error = process_file(file)
    if error:
        return render_template('upload_page.html', error=error)

    # Pass the data as JSON to the loading screen
    return redirect(url_for('loading_screen', dataframe=df.to_json()))


@app.route('/loading', methods=['GET'])
def loading_screen():
    """Displays a loading screen and runs the scheduler."""
    df_json = request.args.get('dataframe')
    if not df_json:
        return render_template('landing_page.html', error="No data passed.")

    try:
        df = pd.read_json(df_json)
    except ValueError as e:
        return render_template('upload_page.html', error=f"Error parsing data: {e}")

    # Run the scheduler
    img_buf, stats, plots, error = run_scheduler(df)
    if error:
        return render_template('upload_page.html', error=error)

    # Convert image buffer to base64 for rendering in template
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return render_template('dashboard.html', stats=stats, img_buf=img_base64, plots=plots)


def validate_file(request):
    """Validates the uploaded file."""
    if 'file' not in request.files:
        return None, "No file uploaded."
    file = request.files['file']
    if file.filename == '':
        return None, "No file selected."
    if not file.filename.endswith(('.xlsx', '.xls')):
        return None, "Invalid file type. Please upload an Excel file (.xlsx or .xls)."
    return file, None


def process_file(file):
    """Processes the uploaded Excel file."""
    try:
        df = pd.read_excel(file)
        if df.empty or not all(col in df.columns for col in REQUIRED_COLUMNS):
            return None, "Invalid or empty Excel file."
        return df, None
    except Exception as e:
        return None, f"Error reading Excel file: {e}"


def generate_plots(stats):
    plots = {}

    # Machine Utilization Plot
    utilization = stats['utilization_rates']
    plt.figure(figsize=(8, 6))
    plt.bar(utilization.keys(), utilization.values(), color='skyblue')
    plt.xlabel("Machine")
    plt.ylabel("Utilization (%)")
    plt.title("Machine Utilization")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['machine_utilization'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Lateness Distribution Plot
    lateness = stats['lateness']
    plt.figure(figsize=(8, 6))
    plt.hist(list(lateness.values()), bins=range(0, max(lateness.values()) + 5), color='salmon', edgecolor='black')
    plt.xlabel("Lateness (Minutes)")
    plt.ylabel("Number of Jobs")
    plt.title("Job Lateness Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['lateness_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Idle Times Plot
    idle_times = stats['idle_times']
    plt.figure(figsize=(8, 6))
    plt.bar(idle_times.keys(), idle_times.values(), color='lightgreen')
    plt.xlabel("Machine")
    plt.ylabel("Idle Time (Minutes)")
    plt.title("Idle Time per Machine")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['idle_times'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return plots


def calculate_utilization(schedule_df, num_machines):
    """Calculate machine utilization based on the actual working time of each machine."""
    utilization_rates = {}
    total_processing_time_per_machine = {m: 0 for m in range(num_machines)}
    start_times = {m: float('inf') for m in range(num_machines)}  # Track the earliest start time
    end_times = {m: 0 for m in range(num_machines)}  # Track the latest end time

    for _, row in schedule_df.iterrows():
        machine = row['Machine']
        processing_time = row['Processing Time']
        start_time = row['Release Time']
        end_time = start_time + processing_time

        # Update total processing time for the machine
        total_processing_time_per_machine[machine] += processing_time

        # Update the earliest start time and latest end time for the machine
        start_times[machine] = min(start_times[machine], start_time)
        end_times[machine] = max(end_times[machine], end_time)

    # Calculate utilization for each machine
    for machine in range(num_machines):
        total_available_time = end_times[machine] - start_times[machine]
        if total_available_time > 0:
            utilization = (total_processing_time_per_machine[machine] / total_available_time) * 100
            utilization_rates[machine] = min(utilization, 100)  # Cap utilization at 100%
        else:
            utilization_rates[machine] = 0  # Avoid division by zero if no available time

    return utilization_rates


def calculate_idle_times(schedule_df, num_machines):
    """Calculate idle times for each machine, including gaps between jobs and idle time before/after jobs."""
    idle_times = {m: 0 for m in range(num_machines)}  # Initialize idle times to zero

    # Group the schedule by machine and sort by start time
    machine_schedules = {m: [] for m in range(num_machines)}
    for _, row in schedule_df.iterrows():
        machine = row['Machine']
        start_time = row['Release Time']
        end_time = start_time + row['Processing Time']
        machine_schedules[machine].append((start_time, end_time))

    # Calculate idle time for each machine
    for machine in range(num_machines):
        schedule = machine_schedules[machine]
        if not schedule:
            # If no jobs are scheduled on this machine, idle time is zero
            idle_times[machine] = 0
            continue

        # Sort the schedule by start time
        schedule.sort()

        # Calculate idle time before the first job
        first_job_start = schedule[0][0]
        idle_before_first_job = first_job_start
        idle_times[machine] += idle_before_first_job

        # Calculate idle time between jobs
        for i in range(1, len(schedule)):
            prev_job_end = schedule[i - 1][1]
            current_job_start = schedule[i][0]
            gap = current_job_start - prev_job_end
            if gap > 0:
                idle_times[machine] += gap

        # Calculate idle time after the last job
        last_job_end = schedule[-1][1]
        total_end_time = max([end for _, end in schedule])  # Total end time of the system
        idle_after_last_job = total_end_time - last_job_end
        idle_times[machine] += idle_after_last_job

    return idle_times

def calculate_lateness(schedule_df, due_dates):
    """Calculate lateness for each job (only positive lateness)."""
    lateness = {}
    for job, due_date in due_dates.items():
        job_schedule = schedule_df[schedule_df['Job'] == job]
        completion_time = job_schedule['Release Time'].max() + job_schedule['Processing Time'].sum()
        lateness[job] = max(0, completion_time - due_date)  # Only consider positive lateness
    return lateness


def calculate_completion_times(schedule_df, num_machines):
    """Calculate the completion time for each job at the final machine."""
    completion_times = {}
    for job in schedule_df['Job'].unique():
        job_schedule = schedule_df[schedule_df['Job'] == job]
        completion_time = job_schedule['Release Time'].max() + job_schedule['Processing Time'].sum()
        completion_times[job] = completion_time
    return completion_times

def run_scheduler(df):
    """Runs the scheduler and returns visualization and stats."""
    try:
        num_jobs = df['Job'].nunique()
        num_machines = df['Machine'].nunique()

        jobs_data = []
        for job in range(num_jobs):
            job_data = []
            for machine in range(num_machines):
                job_machine_data = df[(df['Job'] == job) & (df['Machine'] == machine)]
                if not job_machine_data.empty:
                    processing_time = job_machine_data['Processing Time'].values[0]
                    job_data.append((machine, processing_time))
            jobs_data.append(job_data)

        weights = df.groupby('Job')['Weight'].first().to_dict()
        due_dates = df.groupby('Job')['Due Date'].first().to_dict()
        release_times = df.groupby('Job')['Release Time'].first().to_dict()

        scheduler = ORToolsScheduler(jobs_data, weights, due_dates, release_times, num_jobs, num_machines)
        schedule_df, objective_value = scheduler.solve()

        # Ensure schedule_df is a DataFrame
        if not isinstance(schedule_df, pd.DataFrame):
            return None, None, None, "Schedule data is not a DataFrame."

        # Visualize schedule
        img_buf = io.BytesIO()
        fig = scheduler.visualize_schedule(schedule_df)
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Calculate machine utilization rates
        utilization_rates = calculate_utilization(schedule_df, num_machines)

        # Calculate lateness for each job
        lateness = calculate_lateness(schedule_df, due_dates)

        # Calculate idle times per machine
        idle_times = calculate_idle_times(schedule_df, num_machines)

        # Calculate completion times for each job
        completion_times = calculate_completion_times(schedule_df, num_machines)

        # Prepare stats
        stats = {
            'total_jobs': num_jobs,
            'total_processing_time': df['Processing Time'].sum(),
            'average_processing_time': round(df['Processing Time'].mean(), 2),
            'objective_value': objective_value,
            'utilization_rates': utilization_rates,
            'lateness': lateness,
            'idle_times': idle_times,
            'completion_times': completion_times,  # Add completion times
            'due_dates': due_dates,  # Add due dates
            'release_times': release_times  # Add release times
        }

        # Generate plots
        plots = generate_plots(stats)

        return img_buf, stats, plots, None
    except Exception as e:
        return None, None, None, str(e)

if __name__ == "__main__":
    app.run(debug=True)
