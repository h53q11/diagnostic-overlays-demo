
from pipeline.utils.seed import set_seed
from pipeline.training.train_mlp import run_mlp_demo
from pipeline.training.train_gru import run_gru_demo
from pipeline.attribution.demo import run_attribution_demo

def main():
    set_seed(0)
    print("Running demonstrator pipeline...")

    run_mlp_demo()
    run_gru_demo()
    run_attribution_demo()

if __name__ == "__main__":
    main()
