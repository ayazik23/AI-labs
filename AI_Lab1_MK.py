import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# dataset
df = pd.read_csv(r'data.csv')
df = df.fillna('-')

# Grouping functions
def striking_accuracy_group(acc):
    if acc < 40:
        return 'Low Striking Accuracy'
    elif acc <= 60:
        return 'Medium Striking Accuracy'
    else:
        return 'High Striking Accuracy'

def takedown_accuracy_group(acc):
    if acc < 40:
        return 'Low Takedown Accuracy'
    elif acc <= 60:
        return 'Medium Takedown Accuracy'
    else:
        return 'High Takedown Accuracy'

def stance_group(stance):
    known_stances = ['Orthodox', 'Southpaw', 'Switch']
    return stance if stance in known_stances else 'Other Stance'

def age_group(dob):
    try:
        year_of_birth = int(dob.split('-')[0]) if '-' in dob else 2000
    except:
        year_of_birth = 2000
    age = datetime.now().year - year_of_birth
    return 'Young Fighter' if age < 30 else 'Old Fighter'

df['striking_accuracy_group'] = df['significant_striking_accuracy'].apply(striking_accuracy_group)
df['takedown_accuracy_group'] = df['takedown_accuracy'].apply(takedown_accuracy_group)
df['stance_group'] = df['stance'].apply(stance_group)
df['age_group'] = df['date_of_birth'].apply(age_group)

# Filtering Function
def filter_fighters():
    striking_accuracy = striking_accuracy_var.get()
    takedown_accuracy = takedown_accuracy_var.get()
    stance = stance_var.get()
    age_group = age_group_var.get()

    filtered_df = df.copy()
    if striking_accuracy != "All":
        filtered_df = filtered_df[filtered_df['striking_accuracy_group'] == striking_accuracy]
    if takedown_accuracy != "All":
        filtered_df = filtered_df[filtered_df['takedown_accuracy_group'] == takedown_accuracy]
    if stance != "All":
        filtered_df = filtered_df[filtered_df['stance_group'] == stance]
    if age_group != "All":
        filtered_df = filtered_df[filtered_df['age_group'] == age_group]

    update_table(filtered_df)

# Search Function
def search_fighters():
    query = search_entry.get().strip()
    if not query:
        messagebox.showwarning("Enter a search query")
        return
    
    filtered_df = df[(df['name'].str.contains(query, case=False)) | 
                     (df['nickname'].str.contains(query, case=False))]
    
    if filtered_df.empty:
        messagebox.showinfo("No fighters found!")
    else:
        update_table(filtered_df)

# Update Table
def update_table(dataframe):
    for row in table.get_children():
        table.delete(row)
    for _, row in dataframe.iterrows():
        table.insert("", "end", values=(row['name'], row['nickname'], row['striking_accuracy_group'],
                                         row['takedown_accuracy_group'], row['stance_group'], row['age_group']))

# ML Model Training
def train_model():
    features = ['significant_striking_accuracy', 'takedown_accuracy']
    df_ml = df.copy()
    df_ml['stance_encoded'] = df_ml['stance_group'].astype('category').cat.codes
    df_ml['age_encoded'] = df_ml['age_group'].astype('category').cat.codes
    X = df_ml[features + ['stance_encoded', 'age_encoded']]
    y = df_ml['striking_accuracy_group'].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    messagebox.showinfo("Training Complete", f"Model Accuracy: {accuracy:.2f}")

# GUI
root = tk.Tk()
root.title("Fighter Search")
root.geometry("800x500")

# Filters
frame_filters = tk.Frame(root)
frame_filters.pack(pady=10)

tk.Label(frame_filters, text="Striking Accuracy:").grid(row=0, column=0)
striking_accuracy_var = ttk.Combobox(frame_filters, values=["All", "Low Striking Accuracy", "Medium Striking Accuracy", "High Striking Accuracy"])
striking_accuracy_var.grid(row=0, column=1)
striking_accuracy_var.current(0)

tk.Label(frame_filters, text="Takedown Accuracy:").grid(row=0, column=2)
takedown_accuracy_var = ttk.Combobox(frame_filters, values=["All", "Low Takedown Accuracy", "Medium Takedown Accuracy", "High Takedown Accuracy"])
takedown_accuracy_var.grid(row=0, column=3)
takedown_accuracy_var.current(0)

tk.Label(frame_filters, text="Stance:").grid(row=1, column=0)
stance_var = ttk.Combobox(frame_filters, values=["All", "Orthodox", "Southpaw", "Switch", "Other Stance"])
stance_var.grid(row=1, column=1)
stance_var.current(0)

tk.Label(frame_filters, text="Age Group:").grid(row=1, column=2)
age_group_var = ttk.Combobox(frame_filters, values=["All", "Young Fighter", "Old Fighter"])
age_group_var.grid(row=1, column=3)
age_group_var.current(0)

tk.Button(frame_filters, text="Filter", command=filter_fighters).grid(row=2, column=0, columnspan=4, pady=5)

# Search Box
frame_search = tk.Frame(root)
frame_search.pack(pady=5)

tk.Label(frame_search, text="Search by Name/Nickname:").pack(side=tk.LEFT)
search_entry = tk.Entry(frame_search)
search_entry.pack(side=tk.LEFT, padx=5)
tk.Button(frame_search, text="Search", command=search_fighters).pack(side=tk.LEFT)

# Table
frame_table = tk.Frame(root)
frame_table.pack(pady=10, fill="both", expand=True)

columns = ("Name", "Nickname", "Striking Accuracy", "Takedown Accuracy", "Stance", "Age Group")
table = ttk.Treeview(frame_table, columns=columns, show="headings", height=10)

for col in columns:
    table.heading(col, text=col)
    table.column(col, width=100)

table.pack(fill="both", expand=True)

# Train Model Button
tk.Button(root, text="Train Model", command=train_model).pack(pady=10)

# Run
root.mainloop()
