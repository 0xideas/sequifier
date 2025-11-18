import pandas as pd


def split_file_up(
    data: pd.DataFrame,
    number_of_files: int,
    seq_length: int,
    prediction_length: int,
    folder_path: str,
):
    rows_per_file, leading_rows = divmod(data.shape[0], number_of_files)
    input(f"{rows_per_file = }, {leading_rows = }")

    n_zeros = len(str(number_of_files - 1))
    i_offset = 0
    first_start = True
    for i in range(number_of_files):
        if (i * rows_per_file) >= (seq_length - prediction_length):
            if first_start:
                start = 0
                first_start = False
            else:
                start = (
                    leading_rows + (i * rows_per_file) - seq_length + prediction_length
                )

            end = leading_rows + ((i + 1) * rows_per_file) + 1
            print(
                f"{start = }, {end = }, {(start + (seq_length - prediction_length)) = }, {(end - (start + (seq_length - prediction_length))) = })"
            )
            data_subset = data.iloc[start:end, :]
            data_subset.to_parquet(
                f"{folder_path}/test-inf{str(i-i_offset).zfill(n_zeros)}.parquet"
            )
        else:
            i_offset = int(i)
