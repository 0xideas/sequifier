# Task Log

### Step 1: Foundational Changes in `infer.py`
- [ ] **Change Data Ingestion**: Modify `read_data` to use `pl.scan_parquet()` for lazy loading.
- [ ] **Adapt Data Subsetting**: Update `subset_to_selected_columns` to use Polars' `.filter()` expression.

### Step 2: Refactor Autoregressive Data Expansion
- [ ] **Identify Last Observation per Sequence**: Use Polars' window functions (`.last().over("sequenceId")`).
- [ ] **Generate Future Rows**: Use `pl.int_range` and `explode` to create future timesteps.
- [ ] **Combine and Sort**: Use `pl.concat` to merge original and future data.

### Step 3: Overhaul the Main Autoregressive Loop
- [ ] **Eliminate Helper Dictionaries**: Remove `sequence_id_to_subsequence_ids` and `ids_to_row`.
- [ ] **Adapt the Loop Logic**:
    - [ ] Use a Polars DataFrame for the main `data` object.
    - [ ] Replace `data.loc` with `data.filter`.
    - [ ] Adapt `numpy_to_pytorch` to accept a Polars DataFrame.
- [ ] **Replace `fill_in_predictions`**: Substitute with a new Polars-native function.

### Step 4: Design a New, Polars-Native Prediction Filling Function
- [ ] **Create a Predictions DataFrame**: Convert model output to a Polars DataFrame.
- [ ] **Identify Update Locations with a Join**: Use `.join()` to map predictions to future rows.
- [ ] **Pivot the Updates**: Use `.pivot()` to create a wide-format update DataFrame.
- [ ] **Apply Updates Efficiently**: Use `.update()` to apply the changes.

### Step 5: Finalize Output and I/O
- [ ] **Assemble Final Predictions**: Construct the final `preds` object from the Polars DataFrame.
- [ ] **Adapt `write_data`**: Update `write_data` to use Polars' native `.write_csv()` or `.write_parquet()`.
