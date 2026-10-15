# Author 500-um spatial diversity; only sources named in the audited snapshot.
args <- commandArgs(trailingOnly=TRUE)
runtime <- jsonlite::fromJSON(args[1],simplifyVector=FALSE)
out <- args[2]
snapshot_source <- function(path) {
 actual <- normalizePath(path,mustWork=TRUE)
 root <- paste0(normalizePath(runtime$source_snapshot,mustWork=TRUE),"/")
 stopifnot(startsWith(actual,root),actual %in% unlist(runtime$allowed_sources))
 base::source(actual,local=.GlobalEnv)
}
suppressPackageStartupMessages(library(SpatialExperiment))
stopifnot(as.character(packageVersion("entropy"))=="1.3.2")
snapshot_source(runtime$official_utils)
snapshot_source(runtime$official_spatial)
.CellSPARenvir <- new.env()
.CellSPARenvir[["cell_level_baseline_metrics"]] <- c("total_transciprts","total_genes","cell_area")
ann <- read.csv(file.path(out,"paper_aligned_v1/annotation.csv"))
qc <- read.csv(file.path(out,"paper_aligned_v1/qc_all_cells.csv"))
coord <- read.csv(file.path(out,"paper_aligned_v1/own_cell_centroids.csv"))
structure <- read.csv(file.path(out,"structure/cells_distribution.csv"))
stopifnot(!anyDuplicated(ann$cell_id),!anyDuplicated(qc$cell_id),!anyDuplicated(coord$cell_id),!anyDuplicated(structure$cell_id))
qc <- qc[match(ann$cell_id,qc$cell_id),]
coord <- coord[match(ann$cell_id,coord$cell_id),]
structure <- structure[match(ann$cell_id,structure$cell_id),]
stopifnot(all(qc$qc_pass),all(qc$cell_id==ann$cell_id),all(coord$cell_id==ann$cell_id),all(structure$cell_id==ann$cell_id))
df <- data.frame(cell_id=ann$cell_id,cell_type=ann$evaluation_label,
                 total_transciprts=qc$total_transcripts,total_genes=qc$total_genes,
                 cell_area=structure$area_um2,x_um=coord$x_um,y_um=coord$y_um)
stopifnot(all(is.finite(as.matrix(df[,3:7]))),all(df[,3:5]>0))
spe <- SpatialExperiment::SpatialExperiment(colData=S4Vectors::DataFrame(df),
               spatialCoords=as.matrix(df[,c("x_um","y_um")]))
spe <- .initialise_CellSPA_list(spe)
spe <- calSpatialMetricsDiversity(spe,celltype="cell_type",bin_width=500)
result <- spe@metadata$CellSPA$spatialMetricsDiversity
result$results$n_cells <- as.integer(table(spe$bin_groups)[as.character(result$results$bin)])
valid_cv <- sum(complete.cases(result$results[,c("entropy","cv_total_transciprts","cv_total_genes","cv_cell_area")]))
valid_sd <- sum(complete.cases(result$results[,c("entropy","sd_total_transciprts","sd_total_genes","sd_cell_area")]))
write.csv(result$results,file.path(out,"paper_aligned_v1/spatial_diversity_bins.csv"),row.names=FALSE,na="NA")
write.csv(data.frame(metric=names(result$statistics),pearson=unname(result$statistics),
                     n_qc_cells=nrow(df),n_bins=nrow(result$results),n_bins_complete_cv=valid_cv,n_bins_complete_sd=valid_sd,bin_width=500),
          file.path(out,"paper_aligned_v1/spatial_diversity_correlations.csv"),row.names=FALSE,na="NA")
writeLines(capture.output(sessionInfo()),file.path(out,"spatial_R_sessionInfo.txt"))
writeLines('succeeded',file.path(out,'spatial_execution_status.txt'))
