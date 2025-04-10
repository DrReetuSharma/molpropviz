import os
import time
import pandas as pd
import matplotlib
# Set the matplotlib backend to Agg (non-GUI backend)
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from flask import Flask, request, render_template, send_file, url_for

# Initialize Flask App
app = Flask(__name__)

# Define Folders
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
PLOTS_FOLDER = "plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Molecular Property Functions
PROPERTY_FUNCTIONS = {
    "Molecular Weight": Descriptors.MolWt,
    "Number of Rings": Descriptors.RingCount,
    "Topological Polar Surface Area": Descriptors.TPSA,
    "LogP": Descriptors.MolLogP,
    "Number of Hydrogen Acceptors": Descriptors.NumHAcceptors,
    "Number of Hydrogen Donors": Descriptors.NumHDonors
}

@app.route("/")
def index():
    return render_template("index.html", properties=PROPERTY_FUNCTIONS.keys())

@app.route("/calculate", methods=["POST"])
def calculate_properties():
    start_time = time.time()
    file = request.files.get("file")
    properties = request.form.getlist("properties")
    
    if not file or not properties:
        return render_template("index.html", error="Missing file or properties.")
    
    invalid_props = set(properties) - PROPERTY_FUNCTIONS.keys()
    if invalid_props:
        return render_template("index.html", error=f"Invalid properties selected: {invalid_props}")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    df = pd.read_csv(file_path)
    if "SMILES" not in df.columns or "Database_ID" not in df.columns:
        return render_template("index.html", error="CSV must contain 'Database_ID' and 'SMILES' columns.")
    
    results = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        result = {"Database_ID": row["Database_ID"], "SMILES": row["SMILES"]}
        if mol:
            for prop in properties:
                result[prop] = round(PROPERTY_FUNCTIONS[prop](mol), 2)
        else:
            for prop in properties:
                result[prop] = None
        results.append(result)
    
    result_df = pd.DataFrame(results)
    result_file = os.path.join(RESULTS_FOLDER, "results.csv")
    result_df.to_csv(result_file, index=False)

    plot_files = []
    for prop in properties:
        plt.figure(figsize=(8, 6))
        sns.histplot(result_df[prop].dropna(), bins=20, kde=True)
        plt.xlabel(prop)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {prop}")
        plot_path = os.path.join(PLOTS_FOLDER, f"{prop}.png")
        plt.savefig(plot_path)
        plt.close()
        plot_files.append(f"{prop}.png")
    
    # Send results and plot URLs to index.html
    return render_template("index.html", 
                           results=result_df.to_html(classes="table table-striped"),
                           plot_files=plot_files, 
                           result_file="results.csv", 
                           time_taken_seconds=round(time.time() - start_time, 2))

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return render_template("index.html", error="File not found.")

@app.route("/plot")
def plot():
    prop = request.args.get("property")
    plot_path = os.path.join(PLOTS_FOLDER, f"{prop}.png")
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    return render_template("index.html", error="Plot not found.")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

if __name__ == "__main__":
    app.run(debug=True)
