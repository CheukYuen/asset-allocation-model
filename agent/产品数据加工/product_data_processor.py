import pandas as pd

df_raw = pd.read_csv("raw_products.csv")

columns_needed = [
    "product_code",
    "product_name",
    "primary_strategy",
    "R3",
    "V1",
    "DD2",
    "S3"
]

df = df_raw[columns_needed]
