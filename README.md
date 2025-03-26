# CICIDS2017-dataset-Project
Cybersecurity Threat Classification Using Machine Learning (CIC-IDS2017)
# Cybersecurity Threat Classification Using Machine Learning

## Overview
This project focuses on **classifying cybersecurity threats** using machine learning techniques. The system is trained on the **CICIDS2017** dataset to detect network intrusions effectively. Multiple ML models are trained and evaluated to determine the best approach for threat detection.

## Dataset
The project utilizes the **CICIDS2017** dataset, available at:
[https://www.unb.ca/cic/datasets/ids.html](https://www.unb.ca/cic/datasets/ids.html)

If downloading the dataset is an issue, a small custom dataset with labeled attack categories may be used.

## Task Workflow
1. **Data Preprocessing:**
   - Cleaning missing values
   - Normalization
   - Feature encoding

2. **Feature Selection:**
   - Identifying and extracting relevant features for classification

3. **Model Selection & Training:**
   - Training at least two machine learning models such as:
     - Random Forest
     - Support Vector Machine (SVM)
     - Neural Networks (TensorFlow/PyTorch)

4. **Evaluation:**
   - Model performance comparison using:
     - Accuracy
     - Precision
     - Recall
     - F1-score

5. **Visualization:**
   - Confusion matrix
   - Feature importance plots
   - Other insightful visualizations

## Tech Stack
- **Python**
- **Libraries Used:**
  - `Scikit-learn` (for ML models and evaluation)
  - `Pandas` (for data handling)
  - `NumPy` (for numerical operations)
  - `Matplotlib` & `Seaborn` (for visualization)
  - `TensorFlow/PyTorch` (if deep learning models are used)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cybersecurity-threat-classification.git
   cd cybersecurity-threat-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and prepare the dataset from [CICIDS2017](https://www.unb.ca/cic/datasets/ids.html).
4. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
5. Train the models:
   ```bash
   python train.py
   ```
6. Evaluate the models:
   ```bash
   python evaluate.py
   ```

## Usage
- Modify `config.py` to adjust dataset paths and model parameters.
- Run the scripts sequentially to preprocess data, train models, and evaluate performance.
- Visualize results using provided Jupyter notebooks.

## Results
- Detailed evaluation metrics and visualizations will be saved in the `results/` directory.
- The best-performing model can be used for real-time intrusion detection in a production environment.

## Submission Requirements
- A **Jupyter Notebook** or Python script with the complete implementation.
- A **brief report** (PDF or DOCX, max 3 pages) summarizing the approach, findings, and results.
- A **README file** explaining how to run the code.

## Author
Sri Krishna Mishra  
[GitHub Profile](https://github.com/SriKrishnaMishra/CICIDS2017-dataset-Project/)

## License
This project is licensed under the MIT License.

---
Feel free to contribute or suggest improvements!

