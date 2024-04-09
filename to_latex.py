import pandas as pd


collection_dir = "./experiments/test/fullsup_2024-02"
collection_dir_fewshot = "./experiments/test/pretrained_2024-02"
significant_digits = 3


def main():
    df = pd.read_pickle(f"{collection_dir}.pkl")
    df = df.multiply(100).round(significant_digits - 2).astype(str)
    # rank_df = df.rank(axis=0, ascending=False)
    # avg_rank = rank_df.mean(axis=1).round(1).astype(str)
    # df.insert(0, "average rank", avg_rank)
    styled = df.style.highlight_max(props="bfseries:;")
    latex = styled.to_latex(siunitx=True, column_format="lr" + "U" * len(df.columns), hrules=True)

    print(latex)


def main_fewshot():
    df = pd.read_pickle(f"{collection_dir_fewshot}.pkl")
    mean_df = df["mean auroc"].multiply(100).round(significant_digits - 2)
    ci95_df = df["ci95 auroc"].multiply(10**significant_digits)
    combined_df = (
        mean_df.astype(str) + "(" + ci95_df.round().fillna(0).astype(int).astype(str) + ")"
    )

    combined_df = combined_df[
        combined_df.index.get_level_values("Checkpoint Metric") != "best-loss"
    ]
    combined_df = combined_df.reset_index(level="Checkpoint Metric", drop=True)
    combined_df = combined_df[combined_df.index.get_level_values("Task Loss Scaling") != "False"]
    combined_df = combined_df.reset_index(level="Task Loss Scaling", drop=True)
    combined_df = combined_df[combined_df.index.get_level_values("Fine-tuning Steps") == 50]
    combined_df = combined_df.reset_index(level="Fine-tuning Steps", drop=True)
    combined_df = combined_df[
        combined_df.index.get_level_values("Fine-tuning Learning Rate") == 1e-3
    ]
    combined_df = combined_df.reset_index(level="Fine-tuning Learning Rate", drop=True)

    # drop all columns after mammo_mass
    # combined_df = combined_df[combined_df.columns[: combined_df.columns.get_loc("mammo_mass")]]
    # drop all columns until mammo_mass
    # combined_df = combined_df[combined_df.columns[combined_df.columns.get_loc("mammo_mass") :]]

    styled = combined_df.style.highlight_max(props="bfseries:;")
    latex = styled.to_latex(
        siunitx=True, column_format="@{}lll" + "U" * len(combined_df.columns) + "@{}", hrules=True
    )

    print(latex)


if __name__ == "__main__":
    main_fewshot()
