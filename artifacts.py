import os
import joblib
import pandas as pd

ARTIFACTS_DIR = "artifacts"
OUTPUT_DIR = "artifacts_readable"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_save_pkl(folder_path):
    for file in os.listdir(folder_path):

        if file.endswith(".pkl"):
            file_path = os.path.join(folder_path, file)

            try:
                data = joblib.load(file_path)
                print(f"✅ Loaded: {file}")

                # ----------------------------
                # SAVE BASED ON DATA TYPE
                # ----------------------------

                # 1️⃣ If it's a list → CSV
                if isinstance(data, list):
                    df = pd.DataFrame(data, columns=["values"])
                    df.to_csv(f"{OUTPUT_DIR}/{file}.csv", index=False)

                # 2️⃣ If it's a DataFrame
                elif isinstance(data, pd.DataFrame):
                    data.to_csv(f"{OUTPUT_DIR}/{file}.csv", index=False)

                # 3️⃣ If it's a model / other object
                else:
                    with open(f"{OUTPUT_DIR}/{file}.txt", "w") as f:
                        f.write(str(data))

                print(f"💾 Saved readable version for: {file}")

            except Exception as e:
                print(f"❌ Error with {file}: {e}")


if __name__ == "__main__":
    load_and_save_pkl(ARTIFACTS_DIR)