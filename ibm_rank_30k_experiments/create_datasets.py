import pandas as pd
import argparse
import os


def create_splits(df_path, output_dir):
    df = pd.read_csv(df_path)
    # df['argument_avg_words'] = df["argument"].apply(lambda x: len([len(w) for w in x.split()]))

    df_train = df[df["set"] == "train"]
    df_val = df[df["set"] == "dev"]
    df_test = df[df["set"] == "test"]

    print("Number of train samples: ", len(df_train))
    print("Number of validation samples: ", len(df_val))
    print("Number of testing samples: ", len(df_test))

    df_train.to_csv(
        os.path.join(output_dir, "train_ibm_30k.csv"),
        index=False,
        columns=["argument", "topic", "WA"],
    )
    df_val.to_csv(
        os.path.join(output_dir, "val_ibm_30k.csv"),
        index=False,
        columns=["argument", "topic", "WA"],
    )
    df_test.to_csv(
        os.path.join(output_dir, "test_ibm_30k.csv"),
        index=False,
        columns=["argument", "topic", "WA"],
    )

    return df_train, df_val, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_arg_quality_rank_30k.zip
    parser.add_argument(
        "--full_dataset", default="arg_quality_rank_30k.csv", type=str, required=True
    )
    parser.add_argument("--output_dir", default="datasets", type=str, required=True)
    args = parser.parse_args()

    create_splits(args.full_dataset, args.output_dir)
