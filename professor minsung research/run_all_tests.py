# run_all_tests.py
import importlib

modules = [
    "test_shapes_and_build",
    "test_energy_equivalence_tiny",
    "test_random_spotchecks",
    "test_bitdepth_monotonicity",
    "test_separable_channel",
]

def main():
    for m in modules:
        print(f"Running {m} ...")
        mod = importlib.import_module(m)
        print(f"OK: {m}")

if __name__ == "__main__":
    main()
