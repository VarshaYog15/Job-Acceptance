import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

from config import FEATURE_DATA_PATH


# ============================================================
# MYSQL CONFIGURATION
# ============================================================

MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "job_acceptance_db")

TABLE_NAME = "candidate_data"


# ============================================================
# CREATE DATABASE CONNECTION
# ============================================================

def create_mysql_engine():
    if not MYSQL_PASSWORD:
        raise ValueError(
            "MYSQL_PASSWORD environment variable is not set.\n"
            "Please set your MySQL password before running this script."
        )

    encoded_password = quote_plus(MYSQL_PASSWORD)

    connection_url = (
        f"mysql+pymysql://{MYSQL_USER}:{encoded_password}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    )

    engine = create_engine(connection_url)

    return engine


# ============================================================
# TEST CONNECTION
# ============================================================

def test_connection(engine):

    print("\n🔌 TESTING MYSQL CONNECTION")
    print("-" * 50)

    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()

        print("✅ MySQL connection successful")
        print(f"Database: {MYSQL_DATABASE}")

    except Exception as e:
        print("❌ MySQL connection failed")
        print("Error:", e)
        raise


# ============================================================
# LOAD FEATURE-ENGINEERED DATA
# ============================================================

def load_feature_data():

    print("\n📥 LOADING FEATURE-ENGINEERED DATA")
    print("-" * 50)

    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(
            f"\nFeature-engineered dataset not found:\n"
            f"{FEATURE_DATA_PATH}\n\n"
            "Please run feature_engineering.py first."
        )

    df = pd.read_csv(FEATURE_DATA_PATH)

    print("✅ Dataset loaded successfully")
    print("Shape:", df.shape)

    return df


# ============================================================
# REPLACE MYSQL TABLE
# ============================================================

def save_to_mysql(df, engine):

    print("\n💾 SAVING DATA TO MYSQL")
    print("-" * 50)

    print(f"Table name: {TABLE_NAME}")

    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="replace",
        index=False
    )

    print("✅ Existing table replaced successfully")
    print(f"✅ New table '{TABLE_NAME}' created")
    print(f"Rows inserted: {len(df)}")


# ============================================================
# VERIFY TABLE
# ============================================================

def verify_table(engine):

    print("\n🔍 VERIFYING MYSQL TABLE")
    print("-" * 50)

    with engine.connect() as connection:

        # Count rows
        result = connection.execute(
            text(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        )

        row_count = result.scalar()

        print(f"Total rows in {TABLE_NAME}: {row_count}")

        # Display columns
        result = connection.execute(
            text(f"DESCRIBE {TABLE_NAME}")
        )

        columns = result.fetchall()

        print("\nColumns:")
        for column in columns:
            print(f"  ✓ {column[0]}")

        # Display sample records
        result = connection.execute(
            text(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
        )

        sample_data = result.fetchall()

        print("\nSample records:")
        for row in sample_data:
            print(row)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("🚀 MYSQL DATA LOADING SCRIPT")
    print("=" * 65)

    try:

        # Step 1
        df = load_feature_data()

        # Step 2
        engine = create_mysql_engine()

        # Step 3
        test_connection(engine)

        # Step 4
        save_to_mysql(df, engine)

        # Step 5
        verify_table(engine)

        print("\n" + "=" * 65)
        print("✅ MYSQL DATA LOADING COMPLETED SUCCESSFULLY")
        print("=" * 65)

    except Exception as e:

        print("\n" + "=" * 65)
        print("❌ MYSQL PROCESS FAILED")
        print("=" * 65)

        print("\nError:")
        print(e)


if __name__ == "__main__":
    main()