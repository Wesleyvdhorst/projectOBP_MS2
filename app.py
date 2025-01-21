import random
import io
import base64
from flask import Flask, render_template, request, redirect, url_for
from scheduler import ORToolsScheduler  # Assuming the class exists
import pandas as pd

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
        return render_template('upload_page.html')
    elif choice == 'generate':
        return render_template('generate_data.html')
    return render_template('landing_page.html', error="Invalid selection.")


@app.route('/generate_data', methods=['GET', 'POST'])
def generate_data():
    if request.method == 'GET':
        return render_template('generate_data.html')

    try:
        num_jobs = int(request.form['num_jobs'])
        num_machines = int(request.form['num_machines'])

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
        return redirect(url_for('loading_screen', dataframe=df.to_json()))

    except ValueError:
        return render_template('generate_data.html', error="Please provide valid numerical inputs.")


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and redirects to loading page."""
    file, error = validate_file(request)
    if error:
        return render_template('upload_page.html', error=error)

    df, error = process_file(file)
    if error:
        return render_template('upload_page.html', error=error)

    return redirect(url_for('loading_screen', dataframe=df.to_json()))


@app.route('/loading', methods=['GET'])
def loading_screen():
    """Displays a loading screen and runs the scheduler."""
    df_json = request.args.get('dataframe')
    df = pd.read_json(df_json)

    img_buf, stats, error = run_scheduler(df)
    if error:
        return render_template('upload_page.html', error=error)

    # Convert image buffer to base64 for rendering in template
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return render_template('dashboard.html', stats=stats, img_buf=img_base64)


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


def run_scheduler(df):
    """Runs the scheduler and returns visualization and stats."""
    try:
        num_jobs = df['Job'].nunique()
        num_machines = df['Machine'].nunique()

        jobs_data = [
            [(machine, df[(df['Job'] == job) & (df['Machine'] == machine)]['Processing Time'].values[0])
             for machine in range(num_machines)]
            for job in range(num_jobs)
        ]

        weights = df.groupby('Job')['Weight'].first().values.tolist()
        due_dates = df.groupby('Job')['Due Date'].first().values.tolist()
        release_times = df.groupby('Job')['Release Time'].first().values.tolist()

        scheduler = ORToolsScheduler(jobs_data, weights, due_dates, release_times, num_jobs, num_machines)
        schedule_df, objective_value = scheduler.solve()

        img_buf = io.BytesIO()
        fig = scheduler.visualize_schedule(schedule_df)
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)

        stats = {
            'total_jobs': num_jobs,
            'total_processing_time': df['Processing Time'].sum(),
            'average_processing_time': df['Processing Time'].mean(),
            'objective_value': objective_value
        }
        return img_buf, stats, None
    except Exception as e:
        return None, None, str(e)


if __name__ == "__main__":
    app.run(debug=True)
