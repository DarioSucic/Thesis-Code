import os, json
from pathlib import Path
from typing import List

# os.system("cargo bench")

class Statistic:
    name: str
    confidence_level: float
    lower_bound: float
    upper_bound: float

    def __init__(self, name, data):
        self.name = name
        conf = data["confidence_interval"]
        self.confidence_level = conf["confidence_level"]
        self.lower_bound = conf["lower_bound"]
        self.upper_bound = conf["upper_bound"]

    def __str__(self):
        return f"Statistic([{self.lower_bound:.2e}, {self.upper_bound:.2e}], p={self.confidence_level})"

class BenchmarkResult:
    name: str
    mean: Statistic
    median: Statistic
    std_dev: Statistic

    def __init__(self, name, estimates):
        self.name = name
        self.mean = Statistic("Mean", estimates["mean"])
        self.median = Statistic("Median", estimates["median"])
        self.std_dev = Statistic("Std-Dev", estimates["std_dev"])

    def __str__(self):
        middle = "\n".join(f"    {s.name:8}: {s}" for s in [self.mean, self.median, self.std_dev])
        return f"BenchmarkResult({self.name}) {{\n{middle}\n}}"

def gen_latex_table(results: List[BenchmarkResult]):
    


root = Path("target/criterion/algorithms")

results: List[BenchmarkResult] = []
for path in root.glob("*"):
    if path.name == "report": continue

    with (path / "new" / "estimates.json").open() as file:
        estimates = json.load(file)
    
    results.append(BenchmarkResult(path.name, estimates))


latex_table = gen_latex_table(results)
print(latex_table)
