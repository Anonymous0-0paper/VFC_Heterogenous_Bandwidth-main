import pandas as pd

file_name = 'missed_deadlines_report_Random_First Choice_WithoutNoise.xlsx'

df = pd.read_excel(file_name)

total_deadline_diff = df['deadline_diff'].sum()

print(f"sum of deadline_diff: {total_deadline_diff}")
