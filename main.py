import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# Paths
# -------------------------------
data_path = "../data/students_data.csv"
output_path = "../output"
reports_path = "../reports"

for path in [output_path, reports_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# -------------------------------
# Load Data
# -------------------------------
if not os.path.exists(data_path):
    print("❌ CSV file not found")
    exit()

df = pd.read_csv(data_path)
print("✅ Data Loaded Successfully")

# -------------------------------
# Calculate Marks
# -------------------------------
df["Total_Marks"] = df["Maths"] + df["Science"] + df["English"]
df["Average_Marks"] = df["Total_Marks"] / 3

# -------------------------------
# Performance Category
# -------------------------------
def performance_category(avg):
    if avg >= 85:
        return "Topper"
    elif avg >= 50:
        return "Pass"
    else:
        return "Fail"

df["Performance"] = df["Average_Marks"].apply(performance_category)

# -------------------------------
# Risk Prediction
# -------------------------------
def risk_level(row):
    if row["Average_Marks"] < 50 or row["Attendance"] < 70:
        return "High Risk"
    elif row["Average_Marks"] < 65:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Level"] = df.apply(risk_level, axis=1)

# -------------------------------
# Machine Learning Prediction
# -------------------------------
X = df[["Attendance"]]
y = df["Average_Marks"]

model = LinearRegression()
model.fit(X, y)

df["Predicted_Average_Marks"] = model.predict(X)

# -------------------------------
# Save Outputs
# -------------------------------
df.to_csv("../output/final_student_performance.csv", index=False)

top_students = df.sort_values(by="Average_Marks", ascending=False).head(3)
weak_students = df[df["Performance"] == "Fail"]

top_students.to_csv("../output/top_students.csv", index=False)
weak_students.to_csv("../output/weak_students.csv", index=False)

print("📁 Output files saved successfully")

# -------------------------------
# Graph Functions
# -------------------------------
def performance_bar_chart():
    color_map = {"Topper": "green", "Pass": "blue", "Fail": "red"}
    colors = df["Performance"].map(color_map)

    plt.figure(figsize=(10, 5))
    plt.bar(df["Name"], df["Average_Marks"], color=colors)
    plt.xlabel("Student Name")
    plt.ylabel("Average Marks")
    plt.title("Student Performance (Topper / Pass / Fail)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../reports/student_performance_colored.png")
    plt.show()

def performance_distribution():
    plt.figure(figsize=(6, 4))
    df["Performance"].value_counts().plot(kind="bar")
    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")
    plt.title("Performance Distribution")
    plt.tight_layout()
    plt.savefig("../reports/performance_distribution.png")
    plt.show()

# -------------------------------
# Menu System
# -------------------------------
while True:
    print("\n📌 STUDENT PERFORMANCE ANALYSIS MENU")
    print("1. View Full Student Data")
    print("2. View Top Students")
    print("3. View Weak Students")
    print("4. View Risk Analysis")
    print("5. Show Performance Bar Chart")
    print("6. Show Performance Distribution Chart")
    print("7. Exit")

    choice = input("Enter your choice (1-7): ")

    if choice == "1":
        print(df)

    elif choice == "2":
        print("\n🏆 Top Students")
        print(top_students[["Name", "Average_Marks"]])

    elif choice == "3":
        print("\n⚠️ Weak Students")
        print(weak_students[["Name", "Average_Marks", "Attendance"]])

    elif choice == "4":
        print("\n🚨 Risk Analysis")
        print(df[["Name", "Average_Marks", "Attendance", "Risk_Level"]])

    elif choice == "5":
        performance_bar_chart()

    elif choice == "6":
        performance_distribution()

    elif choice == "7":
        print("👋 Exiting Program")
        break

    else:
        print("❌ Invalid Choice, Try Again")
