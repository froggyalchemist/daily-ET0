import pandas as pd
from rich.console import Console

path = "/work/home/H.mvelasco/SSPs/daily-ET0/test-result/calculate_ET0_log_20260721T151244Z.csv"
df = pd.read_csv(path)

console = Console()
table = df_to_table(df, show_index=False)
console.print(table)
