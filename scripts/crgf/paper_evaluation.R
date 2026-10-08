# QC/normalization/annotation/official markers for one own-cell count matrix.
args <- commandArgs(trailingOnly=TRUE)
p <- jsonlite::fromJSON(args[1],simplifyVector=FALSE); prepared<-args[2]; native<-args[3]; out<-args[4]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(scClassify))
source(p$official_expression)
stopifnot(as.character(packageVersion('scClassify'))==p$scClassify_version,packageDescription('scClassify')$RemoteSha==p$scClassify_commit)
locked<-read.csv(file.path(prepared,'R_packages.csv'),check.names=FALSE)
installed<-installed.packages()
stopifnot(all(locked$Package %in% installed[,'Package']),identical(as.character(installed[locked$Package,'Version']),as.character(locked$Version)))
set.seed(p$seed)
mat<-read.csv(native,row.names=1,check.names=FALSE); ids<-as.character(mat$cell_id);mat$cell_id<-NULL
x<-t(as.matrix(mat));colnames(x)<-ids
stopifnot(nrow(x)==313L,!anyDuplicated(ids),all(is.finite(x)),all(x>=0),all(x==floor(x)))
total<-colSums(x); genes<-colSums(x>0)
pass<-total>=p$qc$min_transcripts & total<=p$qc$max_transcripts & genes>=p$qc$min_genes
reason<-vapply(seq_along(ids),function(i)paste(c(if(total[i]<p$qc$min_transcripts)'low_transcripts',if(total[i]>p$qc$max_transcripts)'high_transcripts',if(genes[i]<p$qc$min_genes)'low_genes'),collapse=';'),character(1))
write.csv(data.frame(cell_id=ids,total_transcripts=total,total_genes=genes,qc_pass=pass,qc_reason=reason),file.path(out,'qc_all_cells.csv'),row.names=FALSE)
stopifnot(sum(pass)>0)
sce<-SingleCellExperiment(assays=list(counts=x[,pass,drop=FALSE]))
sce<-scater::logNormCounts(sce)
write.csv(data.frame(cell_id=colnames(sce),size_factor=sizeFactors(sce)),file.path(out,'size_factors.csv'),row.names=FALSE)
meta<-readRDS(file.path(prepared,'reference_metadata.rds'));model<-readRDS(file.path(prepared,'scClassify_model.rds'))
stopifnot(all(meta$genes %in% rownames(sce)),length(meta$genes)==307L)
expr<-as.matrix(logcounts(sce)[meta$genes,,drop=FALSE]);ids<-colnames(expr)
# All QC-pass cells; batching is deterministic and static thresholds are frozen.
chunks<-split(seq_along(ids),ceiling(seq_along(ids)/p$prediction_batch_size))
raw<-list();annotations<-list()
for(i in seq_along(chunks)) {
 ix<-chunks[[i]]
 pred<-do.call(scClassify::predict_scClassify,c(list(exprsMat_test=expr[,ix,drop=FALSE],trainRes=model),p$prediction))[[1]]
 stopifnot(length(pred$predRes)==length(ix),nrow(pred$predLabelMat)==length(ix))
 raw[[i]]<-pred
 label<-as.character(pred$predRes); mapped<-unname(meta$map[label]);reject<-is.na(mapped)
 why<-ifelse(is.na(label),'missing_label',ifelse(label=='unassigned','classifier_rejected','hierarchical_nonleaf_or_unmatched'))
 mapped[reject]<-'unassigned'
 annotations[[i]]<-data.frame(cell_id=ids[ix],raw_label=label,evaluation_label=mapped,rejected=reject,unmatched_reason=ifelse(reject,why,''),confidence=NA_real_,confidence_status='not_exposed_by_scClassify_public_API',hierarchy=apply(pred$predLabelMat,1,paste,collapse=' -> '),mapping_version=p$mapping_version)
 cat('predicted',max(ix),'of',length(ids),'\n');flush.console()
}
saveRDS(raw,file.path(out,'scClassify_raw_predictions.rds'))
ann<-do.call(rbind,annotations);rownames(ann)<-NULL
stopifnot(identical(ann$cell_id,ids))
write.csv(ann,file.path(out,'annotation.csv'),row.names=FALSE,na='NA')
markers<-readRDS(file.path(prepared,'markers.rds'))
score_markers<-function(markers) {
 # Official helper cannot assign rownames to its 0x0 intermediate when every
 # cell is unmatched. Explicitly implement its documented all-zero append.
 if(!any(ann$evaluation_label %in% names(markers)))return(data.frame(F1=rep(0,length(ids)),Precision=rep(0,length(ids)),Recall=rep(0,length(ids)),row.names=ids))
 .calculate_marker_purity(expr,ann$evaluation_label,markers,BiocParallel::SerialParam())
}
pe<-score_markers(markers$positive)
ne<-score_markers(markers$negative)
names(pe)<-paste0('positive_',names(pe));names(ne)<-paste0('negative_',names(ne))
metrics<-data.frame(cell_id=ids,cell_type=ann$evaluation_label,pe,ne)
stopifnot(all(is.finite(metrics$positive_F1)),all(is.finite(metrics$negative_F1)))
write.csv(metrics,file.path(out,'markers_per_cell.csv'),row.names=FALSE,na='NA')
summary<-do.call(rbind,lapply(c('__all__',meta$types,'unassigned'),function(t) {
 ix<-if(t=='__all__')rep(TRUE,length(ids))else ann$evaluation_label==t
 do.call(rbind,lapply(names(metrics)[3:8],function(k) {v<-metrics[[k]][ix];data.frame(cell_type=t,metric=k,n_total=length(v),n_finite=sum(is.finite(v)),n_nonfinite=sum(!is.finite(v)),all_cell_mean=if(length(v))mean(v)else NA_real_,finite_mean=if(any(is.finite(v)))mean(v[is.finite(v)])else NA_real_,reason=if(!length(v))'empty_type'else if(any(!is.finite(v)))'computed_P_R_nonfinite_retained'else '')}))
}))
write.csv(summary,file.path(out,'marker_summary.csv'),row.names=FALSE,na='NA')
refmean<-as.matrix(read.csv(file.path(prepared,'reference_mean_logcounts.csv'),row.names=1,check.names=FALSE))
refdet<-as.matrix(read.csv(file.path(prepared,'reference_detection_proportions.csv'),row.names=1,check.names=FALSE))
corall<-cor(expr,refmean)
maxcor<-apply(corall,1,max); maxtype<-colnames(corall)[max.col(corall,ties.method='first')]
write.csv(data.frame(cell_id=ids,max_pearson=maxcor,max_pearson_type=maxtype),file.path(out,'single_cell_max_pearson.csv'),row.names=FALSE)
corrows<-list(); vectors<-list()
for(t in meta$types) {
 ix<-ann$evaluation_label==t;n<-sum(ix)
 sm<-if(n)rowMeans(expr[,ix,drop=FALSE])else rep(NA_real_,nrow(expr))
 sd<-if(n)rowMeans(expr[,ix,drop=FALSE]>0)else rep(NA_real_,nrow(expr))
 for(kind in c('mean_logcounts','detection','official_source_detection_vs_mean')) {
 a<-if(kind=='mean_logcounts')sm else sd;b<-if(kind=='detection')refdet[,t]else refmean[,t]
 valid<-n>0 && all(is.finite(a)) && stats::sd(a)>0 && stats::sd(b)>0
 corrows[[length(corrows)+1]]<-data.frame(cell_type=t,definition=kind,n_cells=n,n_genes=length(a),pearson=if(valid)cor(a,b)else NA_real_,status=if(!n)'empty_type'else if(!valid)'constant_or_nonfinite_vector'else 'computed')
 }
 vectors[[length(vectors)+1]]<-data.frame(gene=rownames(expr),cell_type=t,n_cells=n,spatial_mean_logcounts=sm,reference_mean_logcounts=refmean[,t],spatial_detection=sd,reference_detection=refdet[,t])
}
write.csv(do.call(rbind,corrows),file.path(out,'type_correlations.csv'),row.names=FALSE,na='NA')
write.csv(do.call(rbind,vectors),file.path(out,'type_expression_vectors.csv'),row.names=FALSE,na='NA')
# Retain complete spatial gene domain for CD8B and baseline gene detection.
write.csv(data.frame(gene=rownames(sce),fraction_expressing=rowMeans(counts(sce)>0)),file.path(out,'gene_detection_313.csv'),row.names=FALSE)
bt<-counts(sce)[unlist(p$neighbour$markers),,drop=FALSE]
write.csv(data.frame(cell_id=ids,t(as.matrix(bt))),file.path(out,'neighbour_marker_counts.csv'),row.names=FALSE)
write.csv(data.frame(cell_type=c(meta$types,'unassigned'),n_cells=as.integer(table(factor(ann$evaluation_label,levels=c(meta$types,'unassigned'))))),file.path(out,'type_composition.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'))
writeLines('succeeded',file.path(out,'execution_status.txt'))
