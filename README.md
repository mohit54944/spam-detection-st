Installation

Clone the repository (optional):
If you have the project on GitHub, you can clone it.

git clone [https://github.com/mohit54944/spam-detection-st.git]
cd spam-detection-streamlit-app


Create and activate a virtual environment:
It's highly recommended to use a virtual environment to keep project dependencies isolated.

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate


Install the required libraries:
The requirements.txt file contains all the necessary packages.

pip install -r requirements.txt


Download the Dataset:

Download the spam.csv file from the SMS Spam Collection Dataset on Kaggle.

Place the spam.csv file in the same directory as spam_app.py.

Running the Application

Once the setup is complete, run the following command in your terminal:

streamlit run spam_app.py


Your default web browser will automatically open a new tab with the application running.
