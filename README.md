# AlphaCare Insurance Solutions (ACIS)

The ACIS project is dedicated to advancing risk and predictive analytics within car insurance planning and marketing in South Africa, representing an innovative insurance solution that leverages advanced technology and data analytics. The primary objectives of ACIS are to optimize insurance processes, elevate risk assessment capabilities, and enhance customer experiences, all achieved through the utilization of advanced technologies, specifically predictive modeling and data
analytics.

## Usage Instructions

### Data Version Control (DVC)

This project uses Data Version Control (DVC) to track and manage datasets.

Data Version Control (DVC) is an open-source tool that helps you manage and version control your datasets. It works alongside Git to provide a complete solution for reproducible and collaborative machine learning projects.

To use the datasets in this project with DVC, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/Tsegaye16/AlphaCare-Insurance-Solutions
   cd AlphaCare-Insurance-Solutions
   ```

2. Install the project dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Initialize DVC in your project directory:

   ```
   dvc init
   ```

4. Set up a DVC remote storage. For example, to use a remote storage location, run:

   ```
   dvc remote add -d remote_name storage_location
   ```

5. Pull the data from the remote storage:

   ```
   dvc pull
   ```

   This will download the dataset files associated with the project.

6. You can now access the datasets and use them in your project.
