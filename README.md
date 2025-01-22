# Scheduler Application

This repository contains a Flask-based web application that provides tools for scheduling jobs on machines. Users can upload Excel files or generate random data, run a scheduler to optimize job assignments, and visualize results.

---

## Features
- **File Upload**: Upload an Excel file with job and machine details.
- **Data Generation**: Generate random job and machine data.
- **Job Scheduling**: Schedule jobs using an optimization algorithm.
- **Visualization**: View results and statistics in a dashboard with interactive plots.
- **Excel Export**: Download detailed scheduling data as an Excel file.

---

## Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<projectOBP_MS2>.git
cd <projectOBP_MS2>
```

### 2. Set Up a Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

### 3. Install Requirements
Install all dependencies from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Run the Flask app:
```bash
python app.py
```

### 5. Access the Application
After running the app, navigate to the URL displayed in your terminal, typically:
```
http://127.0.0.1:5000
```

---

## Input Requirements
To upload an Excel file, ensure it contains the following columns:
- **Job**: Unique identifier for each job.
- **Machine**: Machine ID for the job.
- **Processing Time**: Time required to process the job.
- **Release Time**: The earliest time the job can start.
- **Due Date**: Deadline for the job.
- **Weight**: Weight assigned to the job for prioritization.

For data generation, you can specify the number of jobs and machines in the application interface.

---

## Outputs
- **Dashboard**: View scheduling statistics, job details, and visualizations.
- **Excel Export**: Download a file with:
  - Job details (lateness, weighted tardiness, etc.).
  - Machine utilization and idle times.
  - Overall statistics.

---

## Notes
- Ensure all dependencies are installed before running the application.
- Random data generation uses a fixed seed for reproducibility.
- The scheduler is powered by the `ORToolsScheduler` class. Ensure it is properly implemented or included in the same directory.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or create a pull request.

---

