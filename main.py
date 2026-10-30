from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


DATA_DIR = Path(__file__).resolve().parent / "data"


def show_result(title, dataframe):
    """Print a labelled query result without truncating category names."""
    print(f"\n{title}")
    dataframe.show(truncate=False)


def describe_dataframe(name, dataframe):
    """Display a compact confirmation that a CSV was read successfully."""
    print(f"\n{name}: {dataframe.count()} rows")
    dataframe.printSchema()


spark = SparkSession.builder.appName("GoIT-DE-HW-03").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 1. Read each input CSV into its own DataFrame.
users_df = spark.read.option("header", True).option("inferSchema", True).csv(
    str(DATA_DIR / "users.csv")
)
purchases_df = spark.read.option("header", True).option("inferSchema", True).csv(
    str(DATA_DIR / "purchases.csv")
)
products_df = spark.read.option("header", True).option("inferSchema", True).csv(
    str(DATA_DIR / "products.csv")
)

print("\n1. Reading source CSV files")
describe_dataframe("users.csv", users_df)
describe_dataframe("purchases.csv", purchases_df)
describe_dataframe("products.csv", products_df)

# 2. Remove rows containing missing values.
users_clean_df = users_df.dropna()
purchases_clean_df = purchases_df.dropna()
products_clean_df = products_df.dropna()

print("\n2. Removing rows with missing values")
print(f"users: {users_df.count()} -> {users_clean_df.count()} rows")
print(f"purchases: {purchases_df.count()} -> {purchases_clean_df.count()} rows")
print(f"products: {products_df.count()} -> {products_clean_df.count()} rows")

# Prepare purchases with product data. A purchase's cost is quantity * unit price.
# This dataset is deliberately independent of users: stage 3 must retain valid
# purchases even if their user row was removed during user-data cleaning.
purchases_with_products_df = (
    purchases_clean_df.join(products_clean_df, on="product_id", how="inner")
    .withColumn("purchase_amount", F.col("quantity") * F.col("price"))
)

# 3. Total purchase amount by product category.
category_totals_df = (
    purchases_with_products_df.groupBy("category")
    .agg(F.round(F.sum("purchase_amount"), 2).alias("total_purchase_amount"))
    .orderBy(F.desc("total_purchase_amount"))
)
show_result("3. Total purchases by category", category_totals_df)

# 4. Purchase amount by category for users aged 18 through 25 inclusive.
young_purchases_df = purchases_with_products_df.join(
    users_clean_df, on="user_id", how="inner"
).filter(F.col("age").between(18, 25))
young_category_totals_df = (
    young_purchases_df.groupBy("category")
    .agg(F.round(F.sum("purchase_amount"), 2).alias("total_purchase_amount"))
    .orderBy(F.desc("total_purchase_amount"))
)
show_result("4. Purchases by category for users aged 18-25", young_category_totals_df)

# 5. Each category's share of total spending by users aged 18 through 25.
young_total_spending_df = young_purchases_df.agg(
    F.sum("purchase_amount").alias("all_categories_total")
)
young_category_shares_df = (
    young_category_totals_df.crossJoin(young_total_spending_df)
    .withColumn(
        "spending_share_percent",
        F.round(
            F.col("total_purchase_amount") / F.col("all_categories_total") * F.lit(100), 2
        ),
    )
    .select("category", "total_purchase_amount", "spending_share_percent")
    .orderBy(F.desc("spending_share_percent"))
)
show_result("5. Category spending share for users aged 18-25", young_category_shares_df)

# 6. The three categories with the highest spending share for users aged 18 through 25.
top_three_categories_df = young_category_shares_df.limit(3)
show_result("6. Top 3 categories by spending share for users aged 18-25", top_three_categories_df)

spark.stop()
