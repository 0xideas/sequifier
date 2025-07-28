
---
### **Overall Strategy** 🎯

The primary goal is to **leverage Polars' lazy execution and expression-based API** to create a more performant and memory-efficient inference pipeline. We will replace `pandas` operations with `polars` equivalents, focusing on eliminating Python loops and row-by-row logic in favor of vectorized, parallel-friendly transformations. The `data` object will be converted to a Polars DataFrame at the earliest possible moment and remain in that format until the final results are written to disk.

---
### **Step 1: Foundational Changes in `infer.py`**

This step focuses on adapting the main `infer` function to use Polars for data ingestion and orchestration.

1.  **Change Data Ingestion**:
    * Modify the `read_data` call inside the `infer` function. Instead of loading the entire dataset into a pandas DataFrame, use `pl.scan_parquet()`.
    * This will create a **LazyFrame**, deferring actual data loading and computation until absolutely necessary, which significantly improves initial speed and reduces peak memory usage. All subsequent operations will be built as a lazy query plan.

2.  **Adapt Data Subsetting**:
    * The `subset_to_selected_columns` helper function currently filters a pandas DataFrame using boolean indexing on the `inputCol`.
    * This function must be updated to accept a Polars DataFrame/LazyFrame and use the `filter()` expression (`.filter(pl.col("inputCol").is_in(selected_columns))`). This operation will be seamlessly integrated into the lazy query plan.

---
### **Step 2: Refactor Autoregressive Data Expansion**

The `expand_data_by_autoregression` function generates future timesteps. The current pandas implementation is iterative and memory-intensive. This will be refactored into a single, declarative Polars expression.

1.  **Identify Last Observation per Sequence**:
    * Instead of `groupby()...last_observation`, use Polars' window functions (`.last().over("sequenceId")`) to find the final valid observation for each sequence in a single pass.

2.  **Generate Future Rows**:
    * Use `pl.int_range` and `explode` to generate the required number of future `subsequenceId` offsets for each sequence.
    * Construct the future rows by combining the last observation's metadata with placeholder values (`pl.lit(np.inf)`) for the sequence data columns. This avoids the Python loop and `pd.concat` pattern entirely.

3.  **Combine and Sort**:
    * Combine the original data with the newly generated future rows using `pl.concat`. The entire operation will remain lazy until a `.collect()` is called.

---
### **Step 3: Overhaul the Main Autoregressive Loop**

This is the core of the migration, where the iterative processing in `get_probs_preds_autoregression` is replaced with a more efficient Polars-native approach.

1.  **Eliminate Helper Dictionaries**:
    * The `sequence_id_to_subsequence_ids` and `ids_to_row` dictionaries are performance bottlenecks. They will be completely removed.
    * Their logic will be replaced by direct operations on the Polars DataFrame. For example, finding future subsequences will be done with a `.filter()` expression inside the loop.

2.  **Adapt the Loop Logic**:
    * The main `data` object will be a **Polars DataFrame** throughout the loop.
    * At the start of each iteration, the `data.loc[...]` call will be replaced with `data.filter(pl.col("subsequenceIdAdjusted") == subsequence_id)`.
    * The `numpy_to_pytorch` helper will be adapted to accept a Polars DataFrame. Since it only extracts data, the change will be minimal (e.g., using `.to_numpy()`).

3.  **Replace `fill_in_predictions`**:
    * The call to the existing `fill_in_predictions` function will be removed.
    * It will be replaced by a new, Polars-native function (e.g., `fill_in_predictions_pl`), as detailed in the next step. This new function will take the main Polars DataFrame and the model's predictions and return the updated DataFrame.

---
### **Step 4: Design a New, Polars-Native Prediction Filling Function** 🚀

This new function, `fill_in_predictions_pl`, will be the new workhorse and will be designed to be highly performant.

1.  **Create a Predictions DataFrame**:
    * Convert the `preds` dictionary (the output from the model) into a small Polars DataFrame, `preds_df`, with columns like `sequenceId`, `inputCol`, and `prediction`.

2.  **Identify Update Locations with a Join**:
    * Perform a `.join()` between the main `data` DataFrame and `preds_df` on `sequenceId` and `inputCol`. This is the key step: it declaratively maps the current predictions to all future rows where they are needed.

3.  **Pivot the Updates**:
    * Calculate the time `offset` for each update using a `with_columns` expression.
    * Use the `.pivot()` method to transform this long-format list of updates into a wide-format DataFrame, where the column names are the offsets ("1", "2", etc.) and the values are the predictions.

4.  **Apply Updates Efficiently**:
    * Use the Polars `.update()` method to apply the pivoted updates to the main `data` DataFrame. This method is highly optimized for this exact use case and modifies the DataFrame efficiently without costly reallocations.

---
### **Step 5: Finalize Output and I/O**

The last step is to ensure the final predictions are formatted correctly and written to disk efficiently.

1.  **Assemble Final Predictions**:
    * After the autoregressive loop completes, the final `preds` object will be constructed directly from the Polars DataFrame. This will involve filtering for the final subsequence ID for each original sequence.

2.  **Adapt `write_data`**:
    * The `write_data` helper function will be updated to detect if it has been passed a Polars DataFrame.
    * If it receives a Polars DataFrame, it will use the native and highly performant `.write_csv()` or `.write_parquet()` methods.

---
### **Expected Outcome** ✨

By following this plan, the inference pipeline will be transformed from a loop-heavy, iterative process into a declarative, set-based one. This will not only provide the **4-5x overall speedup** but will also **eliminate the per-iteration slowdown**, leading to stable, predictable, and significantly more performant autoregressive inference.
