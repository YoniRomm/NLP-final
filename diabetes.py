import pandas as pd

from jinja2 import Template

template_string = """
{{serialization}}
Does this patient have diabetes? Yes or no?
Answer:
{{ answer_choices }}
"""

template = Template(template_string)
answer_choices = "No ||| Yes"

# Replace 'data.csv' with the actual path to your CSV file
csv_file_path = 'diabetes.csv'

# Read the CSV file and create a DataFrame
data_frame = pd.read_csv(csv_file_path)

# Now you can work with the data in the DataFrame
print(data_frame.head())  # Display the first few rows of the DataFrame

# Initialize an empty list to store the formatted strings
formatted_strings = []

# Iterate over the rows of the DataFrame
for index, row in data_frame.iterrows():
    # Construct the formatted string for the current row
    row_string = ', '.join([f'{column}: {value}' for column, value in row.items()])

    # Append the formatted string to the list
    formatted_strings.append(row_string)

    if index == 10:
        break

# Print the formatted strings
for row_string in formatted_strings:
    filled_template = template.render(serialization=row_string, answer_choices=answer_choices)
    print(filled_template)
