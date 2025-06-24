import streamlit as st
import sqlite3
import pandas as pd

# Connect (creates file if not exists)
conn = sqlite3.connect('audit_logs.db')
cursor = conn.cursor()

# Create table (run once)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        action TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

# Insert an audit log entry
cursor.execute('''
    INSERT INTO audit_log (user, action) VALUES (?, ?)
''', ('alice', 'submitted project details'))

conn.commit()
conn.close()

def get_audit_logs():
    conn = sqlite3.connect('audit_logs.db')
    df = pd.read_sql_query('SELECT * FROM audit_log ORDER BY timestamp DESC', conn)
    conn.close()
    return df

# You can add authentication here to restrict to admin users only
st.header("Audit Logs")

logs_df = get_audit_logs()

# Optional: Add filters/search
if st.checkbox("Filter by user"):
    user = st.text_input("Enter username:")
    if user:
        logs_df = logs_df[logs_df['user'] == user]

# Display as a table
st.dataframe(logs_df)