# Cattle Behavior Analysis

This repository contains code for analyzing cattle behavior using different machine learning models. The models available include **Random Forest, Neural Network, and Gradient Boosting**. The script retrieves data from the **Safectory API**, processes it, and classifies cattle behaviors based on beacon tracking data.

## Installation and Setup

Follow these steps to set up and run the project:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mammadli98/bachelorThesis.git
cd bachelorThesis
```

### 2️⃣ Create and Activate a Virtual Environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Credentials
Edit the **main script** (`bachelorThesis.py`) and enter your Safectory credentials:
```python
email = "your_email@example.com"  # Replace with your email
password = "your_password"        # Replace with your password
```

### 5️⃣ Choose a Model
Uncomment one of the following lines in the script to select a model:
```python
selected_model = "Random Forest"  # Uncomment to use Random Forest
#selected_model = "Neural Network"  # Uncomment to use Neural Network
#selected_model = "Gradient Boosting"  # Uncomment to use Gradient Boosting
```

### 6️⃣ Run the Code
Once setup is complete, execute the script:
```bash
python bachelorThesis.py
```

## Results
- The script will retrieve and process beacon data.
- It will train the selected model and evaluate performance.
- Classification reports and visualizations (e.g., confusion matrices) will be displayed.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions
Pull requests are welcome! Feel free to contribute by improving the code or adding new features.

## Contact
For any issues, please open an issue on GitHub or contact **tmammadli@gmx.de**.

