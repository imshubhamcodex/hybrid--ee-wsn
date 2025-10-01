from proposed.RL_PROPOSED import run_proposed
from leach.LEACH_BASELINE import run_leach
from deec.DEEC_BASELINE import run_deec
from fuzzy.FUZZY_C_MEANS_BASELINE import run_fuzzy


METHODS = {
    'LEACH_BASELINE': run_leach,
    'DEEC_BASELINE': run_deec,
    'FUZZY_C_MEANS_BASELINE': run_fuzzy,
    'RL_PROPOSED': run_proposed,
}

if __name__ == "__main__":
    for method in METHODS:
        print(f"Running method: {method}")
        METHODS[method]()