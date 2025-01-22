import random
import base64
from flask import Flask, render_template, request, redirect, url_for, jsonify
import xlsxwriter
import io
from flask import send_file
import json
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


def convert_stats_for_json(stats):
    """Convert numpy types to Python native types for JSON serialization."""
    converted_stats = {}

    for key, value in stats.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (like utilization_rates, idle_times, etc.)
            converted_stats[key] = {
                str(k): float(v) if hasattr(v, 'dtype') else v
                for k, v in value.items()
            }
        elif hasattr(value, 'dtype'):  # Check if it's a numpy type
            converted_stats[key] = float(value) if 'float' in str(value.dtype) else int(value)
        else:
            converted_stats[key] = value

    return converted_stats


@app.route('/loading', methods=['GET'])
def loading_screen():
    """Displays a loading screen and runs the scheduler."""
    df_json = request.args.get('dataframe')
    if not df_json:
        return render_template('landing_page.html', error="No data passed.")

    try:
        # Convert JSON string to StringIO object to avoid FutureWarning
        df = pd.read_json(io.StringIO(df_json))
    except ValueError as e:
        return render_template('upload_page.html', error=f"Error parsing data: {e}")

    # Run the scheduler
    img_buf, stats, plots, error = run_scheduler(df)
    if error:
        return render_template('upload_page.html', error=error)

    # Convert stats for JSON serialization
    converted_stats = convert_stats_for_json(stats)

    # Convert image buffer to base64 for rendering in template
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return render_template('dashboard.html', stats=converted_stats, img_buf=img_base64, plots=plots)


@app.route('/download_excel', methods=['POST'])
def download_excel():
    """Handle the Excel download request with all scheduling data."""
    try:
        # Get stats data from the form
        stats = json.loads(request.form.get('stats_data'))

        # Create an in-memory output file
        output = io.BytesIO()

        # Create the Excel workbook and add worksheets
        workbook = xlsxwriter.Workbook(output)

        # Sheet 1: Job Details
        job_sheet = workbook.add_worksheet("Job Details")
        # Write headers
        headers = ['Job ID', 'Release Time', 'Due Date', 'Weight', 'Completion Time', 'Lateness', 'Weighted Tardiness']
        for col, header in enumerate(headers):
            job_sheet.write(0, col, header)

        # Write job data
        for row, job_id in enumerate(stats['release_times'].keys(), 1):
            job_sheet.write(row, 0, int(job_id))
            job_sheet.write(row, 1, stats['release_times'][job_id])
            job_sheet.write(row, 2, stats['due_dates'][job_id])
            job_sheet.write(row, 3, stats['weights'][job_id])
            job_sheet.write(row, 4, stats['completion_times'][job_id])
            job_sheet.write(row, 5, stats['lateness'][job_id])
            job_sheet.write(row, 6, stats['lateness'][job_id] * stats['weights'][job_id])

        # Sheet 2: Machine Statistics
        machine_sheet = workbook.add_worksheet("Machine Statistics")
        machine_headers = ['Machine ID', 'Utilization Rate (%)', 'Idle Time (minutes)']
        for col, header in enumerate(machine_headers):
            machine_sheet.write(0, col, header)

        for row, machine_id in enumerate(stats['utilization_rates'].keys(), 1):
            machine_sheet.write(row, 0, int(machine_id))
            machine_sheet.write(row, 1, stats['utilization_rates'][machine_id])
            machine_sheet.write(row, 2, stats['idle_times'][machine_id])

        # Sheet 3: Overall Statistics
        stats_sheet = workbook.add_worksheet("Overall Statistics")
        overall_stats = [
            ('Total Jobs', stats['total_jobs']),
            ('Total Processing Time', stats['total_processing_time']),
            ('Average Processing Time', stats['average_processing_time']),
            ('Total Weighted Tardiness', stats['objective_value'])
        ]

        for row, (metric, value) in enumerate(overall_stats):
            stats_sheet.write(row, 0, metric)
            stats_sheet.write(row, 1, value)

        # Close the workbook
        workbook.close()

        # Seek to the beginning of the output
        output.seek(0)

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='schedule_full_data.xlsx'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 400


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
    """Calculate machine utilization rates based on idle time during active period."""
    utilization_rates = {}

    for machine in range(num_machines):
        machine_schedule = schedule_df[schedule_df['Machine'] == machine].sort_values('Start Time')
        if machine_schedule.empty:
            utilization_rates[machine] = 0
            continue

        # Calculate total active period (from first job start to last job end)
        active_period = machine_schedule['End Time'].max() - machine_schedule['Start Time'].min()

        # Calculate idle time (gaps between jobs)
        idle_time = 0
        for i in range(len(machine_schedule) - 1):
            current_end = machine_schedule.iloc[i]['End Time']
            next_start = machine_schedule.iloc[i + 1]['Start Time']
            if next_start > current_end:
                idle_time += next_start - current_end

        # Calculate utilization as percentage of non-idle time during active period
        if active_period > 0:
            utilization = ((active_period - idle_time) / active_period) * 100
            utilization_rates[machine] = round(utilization, 2)
        else:
            utilization_rates[machine] = 0

    return utilization_rates

def calculate_idle_times(schedule_df, num_machines):
    """Calculate idle times as gaps between active processing periods on each machine."""
    idle_times = {m: 0 for m in range(num_machines)}

    for machine in range(num_machines):
        # Get jobs for this machine sorted by start time
        machine_schedule = schedule_df[schedule_df['Machine'] == machine].sort_values('Start Time')
        if len(machine_schedule) <= 1:
            continue

        # Only calculate gaps between consecutive jobs
        for i in range(len(machine_schedule) - 1):
            current_end = machine_schedule.iloc[i]['End Time']
            next_start = machine_schedule.iloc[i + 1]['Start Time']
            if next_start > current_end:
                idle_times[machine] += next_start - current_end

    return {m: round(time, 2) for m, time in idle_times.items()}


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
            'total_processing_time': schedule_df['End Time'].max(),
            'average_processing_time': round(df['Processing Time'].mean(), 2),
            'objective_value': objective_value,
            'utilization_rates': utilization_rates,
            'lateness': lateness,
            'idle_times': idle_times,
            'completion_times': completion_times,
            'due_dates': due_dates,
            'release_times': release_times,
            'weights': weights,
        }

        # Generate plots
        plots = generate_plots(stats)

        return img_buf, stats, plots, None
    except Exception as e:
        return None, None, None, str(e)

if __name__ == "__main__":
    app.run(debug=True)
