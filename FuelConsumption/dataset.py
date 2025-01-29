import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define parameters
num_records = 100

# Predefined categories for string-based features
sexofdriver = ["Male", "Female"]
agebandofdriver = ["18-25", "26-35", "36-45", "46-60", "60+"]
educationlevel = ["High School", "Undergraduate", "Postgraduate"]
vehicledriverrelation = ["Owner", "Employee", "Family Member"]
driverexperience = ["<1 year", "1-3 years", "3-5 years", "5+ years"]
typeofvehicle = ["Car", "Truck", "Bike", "Bus", "Van"]
ownerofvehicle = ["Private", "Company", "Leased"]
defectofvehicle = ["None", "Brake Failure", "Engine Issue", "Tyre Burst"]
roadsurfacecondition = ["Dry", "Wet", "Snowy", "Icy", "Gravel"]

# Generate random data for each feature
data = {
    "sexofdriver": [random.choice(sexofdriver) for _ in range(num_records)],
    "agebandofdriver": [random.choice(agebandofdriver) for _ in range(num_records)],
    "educationlevel": [random.choice(educationlevel) for _ in range(num_records)],
    "vehicledriverrelation": [random.choice(vehicledriverrelation) for _ in range(num_records)],
    "driverexperience": [random.choice(driverexperience) for _ in range(num_records)],
    "typeofvehicle": [random.choice(typeofvehicle) for _ in range(num_records)],
    "ownerofvehicle": [random.choice(ownerofvehicle) for _ in range(num_records)],
    "defectofvehicle": [random.choice(defectofvehicle) for _ in range(num_records)],
    "roadsurfacecondition": [random.choice(roadsurfacecondition) for _ in range(num_records)],
    "fuelconsumption": [random.randint(5, 20) for _ in range(num_records)],  # Integer values for fuel consumption
    "label": [random.choice([0, 1]) for _ in range(num_records)]  # Binary labels 0 or 1
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("fuel_consumption_classification.csv", index=False)

print("Synthetic dataset created and saved as 'fuel_consumption_classification.csv'")
print(df.head())
