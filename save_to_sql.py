import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

print("🔹 Script started")

# Load data
df = pd.read_csv("job_acceptance_features.csv")
print("🔹 Data loaded:", df.shape)



# Create engine
engine = create_engine("mysql+pymysql://root:Vinvj%405050@localhost/job_acceptance_db")

# Test connection
with engine.connect() as conn:
    print("✅ MySQL connection successful")
    conn.execute(text("SELECT 1"))

# Write data
df.to_sql(
    name="job_acceptance",
    con=engine,
    if_exists="replace",
    index=False
)

print("✅ Table job_acceptance created successfully")
