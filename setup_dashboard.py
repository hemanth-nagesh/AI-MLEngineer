import os

# Create necessary directories for the dashboard
directories = [
    "templates",
    "static"
]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

print("Dashboard directories created successfully!")