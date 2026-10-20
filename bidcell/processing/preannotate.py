import argparse
import collections
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import Config, load_config

np.seterr(divide="ignore", invalid="ignore")

N_ANNOTATION_CHUNKS = 16
PREANNOTATION_CHUNK_GLOB = "preannotations_*.csv"


def json_file_to_pyobj(filename):
    """
    Read json config file
    """

    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def normalise_matrix(matrix):
    x_sums = np.sum(matrix, axis=1)
    matrix = matrix / np.expand_dims(x_sums, -1)
    matrix = np.log1p(matrix)
    return matrix


def process_chunk_corr(
    matrix, dir_output, sc_expr, sc_labels, n_atlas_types, output_path=None
):
    if matrix.shape[0] == 0:
        raise ValueError("Cannot annotate an empty chunk.")

    matrix_out = np.zeros((matrix.shape[0], 4))
    col_names = ["cell_id", "cell_type", "spearman", "cell_type_atlas"]

    # cell_type
    cell_genes_norm = normalise_matrix(matrix[:, 1:])
    res = spearmanr(sc_expr, cell_genes_norm, axis=1)
    corr = res.correlation
    # bottom left section
    corr = corr[n_atlas_types:, :n_atlas_types]

    corr_best = np.max(corr, 1)
    best_i_type = np.argmax(corr, 1)
    predicted_cell_type = [sc_labels[x] for x in best_i_type]

    nan_true = np.isnan(corr_best)
    corr_best = [x if not y else -1 for (x, y) in zip(corr_best, nan_true)]
    best_i_type = [x if not y else -1 for (x, y) in zip(best_i_type, nan_true)]
    predicted_cell_type = [
        x if not y else -1 for (x, y) in zip(predicted_cell_type, nan_true)
    ]

    # cell ID
    matrix_out[:, 0] = matrix[:, 0].copy()

    # cell type
    matrix_out[:, 1] = predicted_cell_type.copy()

    # spearman
    matrix_out[:, 2] = corr_best.copy()

    # cell type atlas
    matrix_out[:, 3] = best_i_type.copy()

    # Save as csv
    df_split = pd.DataFrame(
        matrix_out, index=list(range(matrix_out.shape[0])), columns=col_names
    )
    if output_path is None:
        output_path = os.path.join(
            dir_output, "preannotations_%d.csv" % matrix_out[0, 0]
        )
    df_split.to_csv(output_path, index=False)


def _ids_as_int64(values, source):
    ids_numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    ids_float = ids_numeric.to_numpy(dtype=np.float64)

    if ids_numeric.isna().any() or not np.isfinite(ids_float).all():
        raise RuntimeError(f"{source} contains missing or non-numeric cell IDs.")
    if not np.equal(ids_float, np.floor(ids_float)).all():
        invalid_ids = ids_float[ids_float != np.floor(ids_float)][:10].tolist()
        raise RuntimeError(
            f"{source} contains non-integer cell IDs. Examples: {invalid_ids}"
        )

    int64_info = np.iinfo(np.int64)
    if ((ids_float < int64_info.min) | (ids_float > int64_info.max)).any():
        raise RuntimeError(f"{source} contains cell IDs outside the int64 range.")

    return ids_float.astype(np.int64)


def _id_examples(ids, limit=10):
    return sorted(int(cell_id) for cell_id in ids)[:limit]


def _validate_annotations(cell_df, expected_ids):
    required_columns = {"cell_id", "cell_type"}
    missing_columns = required_columns - set(cell_df.columns)
    if missing_columns:
        raise RuntimeError(
            "Merged annotation results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    actual_ids = _ids_as_int64(cell_df["cell_id"], "Annotation results")
    duplicate_mask = pd.Series(actual_ids).duplicated(keep=False).to_numpy()
    duplicate_ids = set(actual_ids[duplicate_mask].tolist())
    expected_id_set = set(expected_ids.tolist())
    actual_id_set = set(actual_ids.tolist())
    missing_ids = expected_id_set - actual_id_set
    extra_ids = actual_id_set - expected_id_set

    problems = []
    if len(cell_df) != len(expected_ids):
        problems.append(
            f"row count differs (expected {len(expected_ids)}, actual {len(cell_df)})"
        )
    if duplicate_ids:
        problems.append(
            f"duplicate IDs: {len(duplicate_ids)} unique duplicated IDs; "
            f"examples {_id_examples(duplicate_ids)}"
        )
    if missing_ids:
        problems.append(
            f"missing IDs: {len(missing_ids)}; examples {_id_examples(missing_ids)}"
        )
    if extra_ids:
        problems.append(
            f"extra IDs: {len(extra_ids)}; examples {_id_examples(extra_ids)}"
        )
    if problems:
        raise RuntimeError(
            "Annotation completeness validation failed: " + "; ".join(problems)
        )

    cell_df = cell_df.copy()
    cell_df["cell_id"] = actual_ids
    return cell_df.sort_values("cell_id").reset_index(drop=True)


def _write_annotations_atomically(cell_df, output_path):
    temporary_path = output_path + ".tmp"
    try:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        with h5py.File(temporary_path, "w") as h5f:
            h5f.create_dataset("data", data=cell_df["cell_type"].to_numpy())
            h5f.create_dataset("ids", data=cell_df["cell_id"].to_numpy(dtype=np.int64))
        os.replace(temporary_path, output_path)
    except Exception as exc:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to write complete annotations to HDF5: {output_path}"
        ) from exc


def preannotate(config: Config):
    dir_dataset = config.files.data_dir
    expr_dir = os.path.join(dir_dataset, config.files.dir_cgm, "nuclei")
    fp_expr = os.path.join(expr_dir, config.files.fp_expr)

    if not os.path.isfile(fp_expr):
        raise FileNotFoundError(f"Nuclei expression matrix not found: {fp_expr}")

    # Cell expressions - order of gene names (columns) will be in same order as all_gene_names.txt
    df_cells = pd.read_csv(fp_expr, index_col=0)
    if df_cells.empty:
        raise RuntimeError(f"Nuclei expression matrix is empty: {fp_expr}")
    if "cell_id" not in df_cells.columns:
        raise RuntimeError(
            f"Nuclei expression matrix does not contain a cell_id column: {fp_expr}"
        )

    expected_ids = _ids_as_int64(df_cells["cell_id"], "Nuclei expression matrix")
    duplicate_input_mask = pd.Series(expected_ids).duplicated(keep=False).to_numpy()
    if duplicate_input_mask.any():
        duplicate_input_ids = set(expected_ids[duplicate_input_mask].tolist())
        raise RuntimeError(
            "Nuclei expression matrix contains duplicate cell IDs: "
            f"{len(duplicate_input_ids)} unique duplicated IDs; "
            f"examples {_id_examples(duplicate_input_ids)}"
        )
    print(f"Number of cells: {df_cells.shape[0]}")

    # Reference data - no requirement of column orders - ensure same order as df_cells
    df_ref_orig = pd.read_csv(config.files.fp_ref, index_col=0)

    # Ensure the order of genes match
    genes_cells = df_cells.columns[1:].tolist()
    ct_columns = df_ref_orig.columns[-3:].tolist()
    df_ref = df_ref_orig[genes_cells + ct_columns]

    genes_ref = df_ref.columns[:-3]
    if list(genes_cells) != list(genes_ref):
        print(
            "Genes in transcripts but not reference: ",
            list(set(genes_cells) - set(genes_ref)),
        )
        print(
            "Genes in reference but not transcripts: ",
            list(set(genes_ref) - set(genes_cells)),
        )
        print("Check names of genes")
        sys.exit()

    sc_expr = df_ref.iloc[:, :-3].to_numpy()
    n_atlas_types = sc_expr.shape[0]
    sc_labels = df_ref.iloc[:, -3].to_numpy().astype(int)
    # sc_names = df_ref.iloc[:, -2].to_list()

    n_annotation_chunks = N_ANNOTATION_CHUNKS
    if n_annotation_chunks < 1:
        raise ValueError("Number of annotation chunks must be at least 1.")
    n_annotation_chunks = min(n_annotation_chunks, len(df_cells))
    print(f"Number of annotation chunks: {n_annotation_chunks}")

    matrix_all = df_cells.to_numpy().astype(np.float32)
    matrix_all_splits = np.array_split(matrix_all, n_annotation_chunks)
    expected_id_splits = np.array_split(expected_ids, n_annotation_chunks)

    # Remove only temporary CSVs belonging to the preannotation workflow. This
    # also prevents chunks from an older failed run being merged into this run.
    old_chunk_paths = glob.glob(os.path.join(dir_dataset, PREANNOTATION_CHUNK_GLOB))
    for old_chunk_path in old_chunk_paths:
        if os.path.isfile(old_chunk_path):
            os.remove(old_chunk_path)

    expected_chunk_paths = [
        os.path.join(
            dir_dataset,
            (
                f"preannotations_chunk_{chunk_index:04d}_of_"
                f"{n_annotation_chunks:04d}.csv"
            ),
        )
        for chunk_index in range(1, n_annotation_chunks + 1)
    ]

    print("Computing simple annotation")
    for chunk_index, (chunk, expected_chunk_ids, chunk_path) in enumerate(
        zip(matrix_all_splits, expected_id_splits, expected_chunk_paths), start=1
    ):
        first_id = int(chunk[0, 0])
        last_id = int(chunk[-1, 0])
        try:
            process_chunk_corr(
                chunk,
                dir_dataset,
                sc_expr,
                sc_labels,
                n_atlas_types,
                output_path=chunk_path,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Annotation chunk {chunk_index}/{n_annotation_chunks} failed. "
                f"Temporary chunk files were retained for debugging: "
                f"{expected_chunk_paths}"
            ) from exc
        if not os.path.isfile(chunk_path):
            raise RuntimeError(
                f"Annotation chunk {chunk_index}/{n_annotation_chunks} produced "
                f"no result file: {chunk_path}"
            )
        try:
            chunk_df = pd.read_csv(chunk_path)
            if chunk_df.empty:
                raise RuntimeError("chunk result is empty")
            _validate_annotations(chunk_df, expected_chunk_ids)
        except Exception as exc:
            raise RuntimeError(
                f"Annotation chunk {chunk_index}/{n_annotation_chunks} produced "
                f"an invalid result: {chunk_path}"
            ) from exc
        print(
            f"Annotation chunk {chunk_index}/{n_annotation_chunks}: "
            f"{len(chunk)} cells, IDs {first_id}-{last_id}"
        )

    fp_chunks = sorted(glob.glob(os.path.join(dir_dataset, PREANNOTATION_CHUNK_GLOB)))
    expected_chunk_set = set(expected_chunk_paths)
    actual_chunk_set = set(fp_chunks)
    if actual_chunk_set != expected_chunk_set:
        missing_chunks = sorted(expected_chunk_set - actual_chunk_set)
        extra_chunks = sorted(actual_chunk_set - expected_chunk_set)
        raise RuntimeError(
            "Annotation chunk file validation failed: "
            f"expected {len(expected_chunk_paths)}, found {len(fp_chunks)}; "
            f"missing {missing_chunks}; extra {extra_chunks}"
        )

    chunk_frames = []
    for chunk_path in expected_chunk_paths:
        try:
            chunk_df = pd.read_csv(chunk_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read annotation chunk result: {chunk_path}"
            ) from exc
        if chunk_df.empty:
            raise RuntimeError(f"Annotation chunk result is empty: {chunk_path}")
        chunk_frames.append(chunk_df)

    cell_df = pd.concat(chunk_frames, axis=0, ignore_index=True)
    cell_df = _validate_annotations(cell_df, expected_ids)

    fp_annotations = os.path.join(dir_dataset, config.files.fp_nuclei_anno)
    _write_annotations_atomically(cell_df, fp_annotations)

    print(f"Expected annotation rows: {len(expected_ids)}")
    print(f"Actual annotation rows: {len(cell_df)}")
    print(f"Unique annotation IDs: {cell_df['cell_id'].nunique()}")
    print(f"Saved complete annotations to: {fp_annotations}")

    # Clean up
    for fpc in expected_chunk_paths:
        os.remove(fpc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    preannotate(config)
