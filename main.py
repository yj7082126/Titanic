
import pandas as pd

# ================ 0.Data Loading ================
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
full = pd.concat([train, test]).set_index("PassengerId")
full.index = full.index.map(int)
print(full.head())

# ================ 1.Data Pre-Processing ================

# ================ 2.Apply ML Algorithm ================

# ================ 3.Evaluate Model ================