import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, avg
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
from math import radians, sin, cos, sqrt, atan2

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Earthquake Analysis") \
    .getOrCreate()

# Task 1: Load the dataset into a PySpark DataFrame
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .load("earthquake_dataset.csv")

# Task 2: Convert the Date and Time columns into a timestamp column named Timestamp
df = df.withColumn("Timestamp", to_timestamp(col("Date") + " " + col("Time")))

# Task 3: Filter the dataset to include only earthquakes with a magnitude greater than 5.0
df_filtered = df.filter(col("Magnitude") > 5.0)

# Task 4: Calculate the average depth and magnitude of earthquakes for each earthquake type
avg_depth_magnitude = df_filtered.groupBy("Type").agg(avg("Depth").alias("AvgDepth"), avg("Magnitude").alias("AvgMagnitude"))

# Task 5: Implement a UDF to categorize the earthquakes into levels (e.g., Low, Moderate, High) based on their magnitudes
def categorize_magnitude(magnitude):
    if magnitude < 5.0:
        return "Low"
    elif magnitude >= 5.0 and magnitude < 7.0:
        return "Moderate"
    else:
        return "High"

categorize_magnitude_udf = udf(categorize_magnitude, returnType=FloatType())
df_categorized = df.withColumn("MagnitudeCategory", categorize_magnitude_udf(col("Magnitude")))

# Task 6: Calculate the distance of each earthquake from a reference location (e.g., (0, 0))
def calculate_distance(lat, lon, ref_lat=0.0, ref_lon=0.0):
    R = 6371.0  # Radius of the Earth in km
    lat1 = radians(lat)
    lon1 = radians(lon)
    lat2 = radians(ref_lat)
    lon2 = radians(ref_lon)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

calculate_distance_udf = udf(calculate_distance, returnType=FloatType())
df_with_distance = df.withColumn("DistanceFromRef", calculate_distance_udf(col("Latitude"), col("Longitude")))

# Task 7: Visualize the geographical distribution of earthquakes on a world map using appropriate libraries
# You can use libraries like Folium or Matplotlib to visualize the data

# Task 8: Write the final DataFrame to a CSV file
df_with_distance.write \
    .format("csv") \
    .option("header", "true") \
    .save("output_earthquake_analysis.csv")

# Stop the SparkSession
spark.stop()
