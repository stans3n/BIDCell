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
stopifnot(nrow(df)>0L,!anyNA(df$cell_type),all(nzchar(df$cell_type)),
          all(is.finite(as.matrix(df[,3:7]))),all(df[,3:5]>0))
spe <- SpatialExperiment::SpatialExperiment(colData=S4Vectors::DataFrame(df),
               spatialCoords=as.matrix(df[,c("x_um","y_um")]))
spe <- .initialise_CellSPA_list(spe)

# Adapt only the author's dimension simplification and undefined correlations.
# Author sources above remain byte-for-byte unchanged. There is no error-catching
# fallback: malformed inputs or unexpected R errors still fail this stage.
author_spatial_with_NA <- function(df,bin_width=500) {
 metrics <- c("total_transciprts","total_genes","cell_area")
 # Exact author cut() definition, including range expansion and minimum 2 breaks.
 x_bin_groups <- cut(df$x_um,breaks=max(ceiling(diff(range(df$x_um))/bin_width),2))
 y_bin_groups <- cut(df$y_um,breaks=max(ceiling(diff(range(df$y_um))/bin_width),2))
 bin_groups <- paste(x_bin_groups,y_bin_groups,sep="|")
 res <- table(df$cell_type,bin_groups)
 bin_entropy <- apply(res,2,function(x)entropy::entropy(x))
 bins <- data.frame(bin=names(bin_entropy),entropy=bin_entropy)
 bins <- cbind(bins,do.call(rbind,strsplit(as.character(bins$bin),"\\|")))
 colnames(bins) <- c("bin","entropy","x_bin","y_bin")
 bins$x_bin <- factor(bins$x_bin,levels=levels(x_bin_groups))
 bins$y_bin <- factor(bins$y_bin,levels=levels(y_bin_groups))
 # sweep()/drop=FALSE preserve bin-by-type shape for one observed class or bin.
 count_matrix <- matrix(as.numeric(res),nrow=nrow(res),ncol=ncol(res),dimnames=dimnames(res))
 celltype_prop <- t(sweep(count_matrix,2,colSums(count_matrix),"/"))
 colnames(celltype_prop) <- paste("cellTypeProp",colnames(celltype_prop),sep="_")
 bins <- cbind(bins,celltype_prop[as.character(bins$bin),,drop=FALSE])
 stopifnot(nrow(bins)==ncol(res),!anyDuplicated(bins$bin))
 for(metric in metrics) {
  cv <- stats::aggregate(log10(df[[metric]]),list(bin_groups),function(x)sd(x)/mean(x))
  colnames(cv) <- c("bin",paste0("cv_",metric));bins <- merge(bins,cv,by="bin")
  sd_values <- stats::aggregate(df[[metric]],list(bin_groups),function(x)stats::sd(x))
  colnames(sd_values) <- c("bin",paste0("sd_",metric));bins <- merge(bins,sd_values,by="bin")
 }
 bins$n_cells <- as.integer(table(bin_groups)[as.character(bins$bin)])
 for(metric in metrics) {
  log_mean <- tapply(log10(df[[metric]]),bin_groups,mean)[as.character(bins$bin)]
  cv_name <- paste0("cv_",metric);sd_name <- paste0("sd_",metric)
  bins[[paste0(cv_name,"_status")]] <- ifelse(bins$n_cells<2L,"insufficient_cells_for_sample_sd",
       ifelse(log_mean==0,"zero_mean_log10_metric",ifelse(is.finite(bins[[cv_name]]),"computed","nonfinite_statistic")))
  bins[[paste0(sd_name,"_status")]] <- ifelse(bins$n_cells<2L,"insufficient_cells_for_sample_sd",
       ifelse(is.finite(bins[[sd_name]]),"computed","nonfinite_statistic"))
  bins[[cv_name]][!is.finite(bins[[cv_name]])] <- NA_real_
  bins[[sd_name]][!is.finite(bins[[sd_name]])] <- NA_real_
 }
 valid <- list();complete <- list()
 for(family in c("cv","sd")) {
  mat <- as.matrix(bins[,c("entropy",paste(family,metrics,sep="_")),drop=FALSE])
  complete[[family]] <- complete.cases(mat)
  valid[[family]] <- complete[[family]] & apply(is.finite(mat),1,all)
  bins[[paste0(family,"_family_complete_finite")]] <- valid[[family]]
 }
 statistics <- list()
 for(family in c("cv","sd")) {
  # cor(entropy, three-metric matrix, use='complete.obs') uses the SAME rows
  # for all three columns. Do not switch to per-metric pairwise deletion.
  keep <- valid[[family]];n <- sum(keep);x <- bins$entropy[keep]
  for(metric in paste(family,metrics,sep="_")) {
   y <- bins[[metric]][keep]
   reason <- if(n<2L)"fewer_than_two_common_finite_bins" else
             if(stats::sd(x)==0 && stats::sd(y)==0)"constant_entropy_and_metric" else
             if(stats::sd(x)==0)"constant_entropy" else
             if(stats::sd(y)==0)"constant_metric" else ""
   pearson <- if(nzchar(reason))NA_real_ else stats::cor(x,y)
   if(!nzchar(reason) && !is.finite(pearson))stop("unexpected nonfinite Pearson correlation for ",metric)
   statistics[[length(statistics)+1L]] <- data.frame(metric=metric,pearson=pearson,
       status=if(nzchar(reason))"NA" else "computed",reason=reason,family=family,
       n_qc_cells=nrow(df),n_bins=nrow(bins),n_bins_complete_cv=sum(complete$cv),
       n_bins_complete_sd=sum(complete$sd),n_bins_common_finite_cv=sum(valid$cv),
       n_bins_common_finite_sd=sum(valid$sd),n_common_finite_bins=n,
       n_metric_finite_bins=sum(is.finite(bins$entropy)&is.finite(bins[[metric]])),
       n_excluded_by_family=nrow(bins)-n,
       population_rule="family_common_complete_finite_entropy_and_all_three_metrics",
       bin_width=bin_width)
  }
 }
 list(results=bins,statistics=do.call(rbind,statistics))
}
result <- author_spatial_with_NA(df,bin_width=500)
write.csv(result$results,file.path(out,"paper_aligned_v1/spatial_diversity_bins.csv"),row.names=FALSE,na="NA")
write.csv(result$statistics,
          file.path(out,"paper_aligned_v1/spatial_diversity_correlations.csv"),row.names=FALSE,na="NA")
writeLines(capture.output(sessionInfo()),file.path(out,"spatial_R_sessionInfo.txt"))
writeLines('succeeded',file.path(out,'spatial_execution_status.txt'))
