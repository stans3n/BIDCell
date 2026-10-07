# Thin adapter: source pinned official functions unchanged. Engineering profiles only.
args <- commandArgs(trailingOnly=TRUE)
src<-args[1]; native<-args[2]; refpath<-args[3]; pospath<-args[4]; negpath<-args[5]; annotation<-args[6]; out<-args[7]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
suppressPackageStartupMessages(library(SpatialExperiment))
suppressPackageStartupMessages(library(Matrix))
source(file.path(src,'expression_metrics.R'))
source(file.path(src,'neighbor_metrics.R'))
source(file.path(src,'baseline_metrics.R'))
source(file.path(src,'utils.R'))
mat<-read.csv(native,row.names=1,check.names=FALSE);ids<-as.character(mat$cell_id); mat$cell_id<-NULL
expr<-t(as.matrix(mat));colnames(expr)<-ids
ref<-read.csv(refpath,row.names=1,check.names=FALSE)
stopifnot(all(rownames(expr)%in%colnames(ref)))
refexpr<-t(as.matrix(ref[,rownames(expr),drop=FALSE]))
ann<-read.csv(annotation,check.names=FALSE);stopifnot(identical(as.character(ann$cell_id),ids))
cor<- .cal_expression_correlation(expr,refexpr,as.character(ref$cell_type),'correlation')
write.csv(data.frame(cell_id=ids,cor),file.path(out,'official_expression_max_pearson.csv'),row.names=FALSE,na='NA')
markers<-function(path){m<-read.csv(path,row.names=1,check.names=FALSE);setNames(lapply(seq_len(nrow(m)),function(i)colnames(m)[m[i,]>0]),rownames(m))}
pos<-markers(pospath);neg<-markers(negpath)
for(profile in c('remote_nuclear_annotation_raw','windows_style_maxpearson_raw')){
  dest<-file.path(out,profile);dir.create(dest,showWarnings=FALSE)
  tryCatch({
  types<-if(profile=='remote_nuclear_annotation_raw')as.character(ann$cell_type) else as.character(cor$celltype_correlation)
  eligible<-types%in%names(pos)&types%in%names(neg)
  # Missing annotation is not a zero-quality observation. Call official implementation
  # only on its known-type domain, and export unknown cells as NA in a transparent join.
  pe<-.calculate_marker_purity(expr[,eligible,drop=FALSE],types[eligible],pos,BiocParallel::SerialParam())
  ne<-.calculate_marker_purity(expr[,eligible,drop=FALSE],types[eligible],neg,BiocParallel::SerialParam())
  p<-data.frame(F1=rep(NA_real_,length(ids)),Precision=NA_real_,Recall=NA_real_);n<-p
  p[eligible,]<-pe;n[eligible,]<-ne
  names(p)<-paste0('positive_',names(p));names(n)<-paste0('negative_',names(n))
  metrics<-data.frame(cell_id=ids,cell_type=types,p,n)
  write.csv(metrics,file.path(dest,'official_markers_per_cell.csv'),row.names=FALSE,na='NA')
  summaries<-do.call(rbind,lapply(names(metrics)[3:8],function(metric){x<-metrics[[metric]];data.frame(metric=metric,all_cell_mean=mean(x),finite_mean=mean(x[is.finite(x)]),n_total=length(x),n_finite=sum(is.finite(x)),n_nonfinite=sum(!is.finite(x)))}))
  write.csv(summaries,file.path(dest,'official_marker_summary.csv'),row.names=FALSE,na='NA')
  coords<-as.matrix(ann[,c('x_um','y_um')]);rownames(coords)<-ids
  spe<-SpatialExperiment(assays=list(counts=expr),spatialCoords=coords)
  spe<-.initialise_CellSPA_list(spe)
  spe<-.cal_baseline(spe)
  write.csv(data.frame(gene=rownames(expr),as.data.frame(rowData(spe))),file.path(dest,'official_gene_detection.csv'),row.names=FALSE)
  # Named Windows-comparable engineering direction; never creates a scalar or final protocol.
  bt<-ifelse(types=='B','B',ifelse(types%in%c('CD4Tconv/Treg','CD8T/CD8Tex'),'T','OTHER'))
  use<-bt%in%c('B','T'); pair<-spe[,use];btypes<-bt[use]
  tryCatch({
  pair<-calNegMarkerVsDist(pair,btypes,c('B','T'),list(B=c('CD4','CD8A','CD8B')),exprs_values='counts')
  curve<-pair@metadata$CellSPA$negMarkerExprs_vs_dist$B
  write.csv(curve,file.path(dest,'official_B_to_T_curve.csv'),na='NA')
  saveRDS(pair@metadata$CellSPA$negMarkerExprs_vs_dist,file.path(dest,'official_B_to_T_curve.rds'))
  },error=function(e)writeLines(conditionMessage(e),file.path(dest,'curve_failed.txt')))
  },error=function(e)writeLines(conditionMessage(e),file.path(dest,'profile_failed.txt')))
}
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'))
writeLines('succeeded',file.path(out,'execution_status.txt'))
