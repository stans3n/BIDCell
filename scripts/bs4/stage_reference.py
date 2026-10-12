"""Derive stage score availability from saved integer counts, never score values."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.crgf.common import identity,sha256
from scripts.cos.diagnostics import METRICS
from .tables import table_path


def normalize_reference_domain(table, count_ids, counts, genes, reference_genes):
    from .stage_population import population_index
    own_ids=population_index(table.cell_id);count_ids=population_index(count_ids)
    genes=pd.Index(genes);reference_genes=pd.Index(reference_genes)
    if genes.has_duplicates or reference_genes.has_duplicates or genes.isna().any() or reference_genes.isna().any():raise ValueError('duplicate or missing gene name')
    if not len(reference_genes):raise ValueError('empty reference gene inventory')
    columns=genes.get_indexer(reference_genes)
    if np.any(columns<0):raise ValueError('missing named reference gene')
    counts=np.asarray(counts)
    if counts.shape!=(len(count_ids),len(genes)):raise ValueError('stage count ID/gene shape mismatch')
    if not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0):raise ValueError('stage counts must be nonnegative integers')
    if counts.size and int(counts.max())>np.iinfo(np.int64).max//len(reference_genes):raise ValueError('reference count sum overflow')
    indices=count_ids.get_indexer(own_ids)
    if np.any(indices<0):raise ValueError('stage row missing authoritative counts ID')
    zero_all=counts[:,columns].sum(axis=1,dtype=np.int64)==0
    empty=zero_all[indices]
    if 'zero_reference_counts' in table:
        values=table['zero_reference_counts']
        if values.isna().any() or not values.isin([True,False]).all() or not np.array_equal(values.to_numpy(bool),empty):raise ValueError('existing zero-reference flag disagrees with counts')
    corrected=table.copy();changed={m:int((empty & np.isfinite(pd.to_numeric(table[m],errors='raise'))).sum()) for m in METRICS}
    corrected['zero_reference_counts']=empty
    corrected.loc[empty,METRICS]=np.nan
    return corrected,dict(n_count_ids=len(count_ids),n_stage_rows=len(own_ids),n_zero_reference_count_ids=int(zero_all.sum()),n_zero_reference_stage_rows=int(empty.sum()),n_zero_reference_known_rows=int((empty & table.label_availability.eq('known')).sum()),existing_zero_flag='zero_reference_counts' in table,finite_values_replaced_by_NA=changed)


def load_stage_table(method,anchor,stage,prepared):
    folder=Path(method['evaluation'])/'diagnostics/stages'
    tablefile=table_path(folder/f'fixed_{anchor}_{stage}.csv');countfile=folder/f'counts_{stage}.npz'
    table_authority=method.get('stage_table_authority')
    authority_receipt=Path(method.get('stage_table_authority_receipt',Path(method['evaluation'])/'complete.json'))
    if table_authority is None:table_authority=json.loads(authority_receipt.read_text())['outputs']
    matches=[obj for obj in table_authority.values() if Path(obj['path']).resolve()==tablefile.resolve()]
    if len(matches)!=1 or matches[0]['sha256']!=sha256(tablefile):raise ValueError('original fixed stage table authority drift')
    validationfile=folder/'validation.json';validation=json.loads(validationfile.read_text())
    if validation.get('status')!='succeeded' or validation.get('integer_conservation') is not True:raise ValueError('stage count validation incomplete')
    count_identity=identity(countfile);declared=validation['count_artifacts'][stage]
    if Path(declared['path']).resolve()!=countfile.resolve() or declared['sha256']!=count_identity['sha256'] or declared['bytes']!=count_identity['bytes']:raise ValueError('authoritative stage count receipt drift')
    genefile=Path(method['matrix'])/'gene_mapping.csv';mapping=pd.read_csv(genefile)
    matrix_receipt=Path(method['matrix'])/'complete.json'
    matrix=json.loads(matrix_receipt.read_text());gene_authority=matrix['outputs']['gene_mapping.csv']
    if gene_authority['sha256']!=sha256(genefile) or Path(gene_authority['path']).resolve()!=genefile.resolve():raise ValueError('frozen matrix gene mapping drift')
    if 'gene' not in mapping or 'gene_column' not in mapping:raise ValueError('gene mapping schema')
    if mapping.gene_column.tolist()!=list(range(len(mapping))):raise ValueError('gene mapping order drift')
    reffile=Path(prepared)/'reference_genes.txt';refs=reffile.read_text().splitlines()
    prepared_receipt=Path(prepared)/'complete.json';reference_authority=json.loads(prepared_receipt.read_text())['outputs']['reference_genes.txt']
    if reference_authority['sha256']!=sha256(reffile) or Path(reference_authority['path']).resolve()!=reffile.resolve():raise ValueError('prepared reference inventory drift')
    if len(refs)!=307:raise ValueError('expected frozen named 307-reference inventory')
    with np.load(countfile,allow_pickle=False) as saved:
        if set(saved.files)!={'cell_id','counts'}:raise ValueError('stage counts schema')
        table,summary=normalize_reference_domain(pd.read_csv(tablefile),saved['cell_id'],saved['counts'],mapping.gene.tolist(),refs)
    inputs={name:identity(path) for name,path in [('table',tablefile),('counts',countfile),('validation',validationfile),('gene_mapping',genefile),('reference_genes',reffile),('matrix_receipt',matrix_receipt),('prepared_receipt',prepared_receipt),('table_authority_receipt',authority_receipt)]}
    return table,dict(inputs=inputs,summary=summary)
