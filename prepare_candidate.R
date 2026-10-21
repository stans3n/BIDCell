# Generate a reviewed candidate outside prediction_chunks; never publish it.
# Usage: Rscript --vanilla prepare_candidate.R ABS_RUN_ROOT
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
root <- normalizePath(args[1], mustWork = TRUE)
repair_dir <- file.path(root, "repairs/singleton")
out <- file.path(repair_dir, "candidate")
stopifnot(!dir.exists(out))
dir.create(out, recursive = TRUE)
log_event <- function(x) { cat(format(Sys.time(), tz="UTC"), x, "\n"); flush.console() }
log_event("loading frozen R packages")
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(scClassify))
source(file.path(repair_dir, "clone_singleton.R"), local = TRUE)
sha <- function(path) digest::digest(path, algo="sha256", file=TRUE, serialize=FALSE)
objsha <- function(x) digest::digest(x, algo="sha256", serialize=TRUE)
identity <- function(path) list(path=normalizePath(path, mustWork=TRUE),
                               bytes=unname(file.info(path)$size), sha256=sha(path))
cfgpath <- file.path(root, "configs/run.json")
cfg <- jsonlite::fromJSON(cfgpath, simplifyVector=FALSE)
protocol <- cfg$evaluation$protocol; prepared <- cfg$evaluation$prepared
native <- file.path(root, "endpoints/P2_MORPH/matrix/expr_mat.csv")
paper <- file.path(root, "endpoints/P2_MORPH/evaluation/paper_aligned_v1")
chunkdir <- file.path(paper, "prediction_chunks")
runtime_path <- file.path(dirname(paper), "r_runtime.json")
runtime <- jsonlite::fromJSON(runtime_path, simplifyVector=FALSE)
p <- jsonlite::fromJSON(protocol, simplifyVector=FALSE)
stopifnot(as.character(packageVersion("scClassify")) == p$scClassify_version,
          packageDescription("scClassify")$RemoteSha == p$scClassify_commit,
          p$prediction_batch_size == 256L, p$prediction$cutoff_method == "static",
          !p$prediction$parallel, !p$prediction$weighted_ensemble)
locked <- read.csv(file.path(prepared, "R_packages.csv"), check.names=FALSE)
installed <- installed.packages()
stopifnot(all(locked$Package %in% installed[,"Package"]),
          identical(as.character(installed[locked$Package,"Version"]),
                    as.character(locked$Version)))
bound <- c(cfgpath, protocol, native, runtime_path,
           file.path(prepared, c("reference_metadata.rds", "scClassify_model.rds", "R_packages.csv")),
           file.path(root, "source/scripts/rep1_triple_simple_boundary/paper_evaluation.R"),
           file.path(repair_dir, c("clone_singleton.R", "prepare_candidate.R")),
           file.path(paper, c("qc_all_cells.csv", "size_factors.csv")))
inputs_before <- lapply(bound, identity)
stopifnot(!file.exists(file.path(chunkdir, "batch_0302.rds")),
          !file.exists(file.path(chunkdir, "batch_0302.rds.committed.json")))
log_event("reconstructing full 313-gene QC and normalization")
# These computations are identical to frozen paper_evaluation.R; no subset
# normalization and no changes to the 256-cell prediction batch definition.
set.seed(p$seed)
mat <- read.csv(native, row.names=1, check.names=FALSE)
ids <- as.character(mat$cell_id); mat$cell_id <- NULL
x <- t(as.matrix(mat)); colnames(x) <- ids
stopifnot(nrow(x)==313L, !anyDuplicated(ids), all(is.finite(x)),
          all(x>=0), all(x==floor(x)))
total <- colSums(x); genes <- colSums(x>0)
pass <- total>=p$qc$min_transcripts & total<=p$qc$max_transcripts & genes>=p$qc$min_genes
qc <- read.csv(file.path(paper,"qc_all_cells.csv"), check.names=FALSE)
stopifnot(identical(as.character(qc$cell_id), ids), identical(qc$qc_pass, unname(pass)),
          identical(as.numeric(qc$total_transcripts), unname(as.numeric(total))),
          identical(as.numeric(qc$total_genes), unname(as.numeric(genes))), sum(pass)==77057L)
sce <- SingleCellExperiment(assays=list(counts=x[,pass,drop=FALSE]))
sce <- scater::logNormCounts(sce)
# Compare the exact CSV representation, avoiding round-trip decimal tolerance.
sf_check <- file.path(out, "reconstructed_size_factors.csv")
write.csv(data.frame(cell_id=colnames(sce),size_factor=sizeFactors(sce)), sf_check, row.names=FALSE)
stopifnot(identical(sha(sf_check), sha(file.path(paper,"size_factors.csv"))))
meta <- readRDS(file.path(prepared,"reference_metadata.rds"))
model <- readRDS(file.path(prepared,"scClassify_model.rds"))
stopifnot(all(meta$genes %in% rownames(sce)), length(meta$genes)==307L)
expr <- as.matrix(logcounts(sce)[meta$genes,,drop=FALSE]); ids <- colnames(expr)
chunks <- split(seq_along(ids),ceiling(seq_along(ids)/p$prediction_batch_size))
stopifnot(length(chunks)==302L, length(chunks[[302L]])==1L)
log_event("verifying all 301 committed batch hashes, identities and RNG states")
prior <- vector("list",301L)
for (i in seq_len(301L)) {
  f <- file.path(chunkdir,sprintf("batch_%04d.rds",i))
  receipt <- paste0(f,".committed.json")
  saved <- jsonlite::fromJSON(receipt,simplifyVector=FALSE); obj <- readRDS(f)
  stopifnot(identical(saved$dependencies_digest,runtime$dependencies_digest),
            identical(saved$sha256,sha(f)), identical(obj$ids,ids[chunks[[i]]]),
            identical(obj$dependencies_digest,runtime$dependencies_digest),
            is.integer(obj$rng_state), length(obj$rng_state)>1L,
            length(obj$prediction$predRes)==length(chunks[[i]]),
            nrow(obj$prediction$predLabelMat)==length(chunks[[i]]))
  prior[[i]] <- list(batch=i,rds=identity(f),receipt=identity(receipt))
}
batch300 <- readRDS(prior[[300L]]$rds$path)
batch301 <- readRDS(prior[[301L]]$rds$path)
clone <- clone_singleton_predictor()
original_body_hash <- objsha(body(clone$original_single))
writeLines(deparse(clone$original_single, width.cutoff=120L), file.path(out,"original_single.R"))
writeLines(deparse(clone$single, width.cutoff=120L), file.path(out,"patched_single.R"))
run_predict <- function(fn, input, rng) {
  assign(".Random.seed",rng,envir=.GlobalEnv)
  warns <- character()
  value <- withCallingHandlers(tryCatch(
    do.call(fn,c(list(exprsMat_test=input,trainRes=model),p$prediction))[[1]],
    error=function(e)e), warning=function(w) {
      warns <<- c(warns,conditionMessage(w)); invokeRestart("muffleWarning")
    })
  list(value=value,rng=get(".Random.seed",envir=.GlobalEnv),warnings=warns)
}
log_event("testing original and isolated clone on unchanged real batch 301")
multi_input <- expr[,chunks[[301L]],drop=FALSE]
multi_original <- run_predict(scClassify::predict_scClassify,multi_input,batch300$rng_state)
stopifnot(!inherits(multi_original$value,"error"),
          identical(multi_original$value,batch301$prediction),
          identical(multi_original$rng,batch301$rng_state))
multi_clone <- run_predict(clone$predict,multi_input,batch300$rng_state)
stopifnot(!inherits(multi_clone$value,"error"),
          identical(multi_original$value,multi_clone$value),
          identical(multi_original$rng,multi_clone$rng),
          identical(multi_original$warnings,multi_clone$warnings))
log_event("reproducing original singleton failure and testing restored matrix shape")
singleton <- expr[,chunks[[302L]],drop=FALSE]
failed <- run_predict(scClassify::predict_scClassify,singleton,batch301$rng_state)
stopifnot(inherits(failed$value,"error"),
          grepl("dim(X) must have a positive length",conditionMessage(failed$value),fixed=TRUE))
single <- run_predict(clone$predict,singleton,batch301$rng_state)
stopifnot(!inherits(single$value,"error"), identical(single$rng,failed$rng),
          identical(single$warnings,failed$warnings),
          length(single$value$predRes)==1L, nrow(single$value$predLabelMat)==1L,
          identical(rownames(single$value$predLabelMat),colnames(singleton)))
# Test oracle only: two identical normalized columns preserve the original
# per-batch expressed-gene filter. No duplicated cell enters the candidate.
oracle_input <- singleton[,c(1L,1L),drop=FALSE]
colnames(oracle_input) <- paste0(colnames(singleton),c("__oracle_1","__oracle_2"))
oracle <- run_predict(scClassify::predict_scClassify,oracle_input,batch301$rng_state)
stopifnot(!inherits(oracle$value,"error"),
          identical(unname(single$value$predRes),unname(oracle$value$predRes[1L])),
          identical(unname(single$value$predRes),unname(oracle$value$predRes[2L])),
          identical(unname(single$value$predLabelMat),unname(oracle$value$predLabelMat[1L,,drop=FALSE])),
          identical(unname(single$value$predLabelMat),unname(oracle$value$predLabelMat[2L,,drop=FALSE])))
stopifnot(identical(objsha(body(get("predict_scClassifySingle",asNamespace("scClassify")))), original_body_hash))
provenance <- list(repair="isolated_scClassify_vapply_singleton_shape", schema_version=1L,
  scientific_change=FALSE, package_version=as.character(packageVersion("scClassify")),
  package_commit=packageDescription("scClassify")$RemoteSha,
  original_body_sha256=original_body_hash,patched_body_sha256=objsha(body(clone$single)),
  insertion_after_statement=clone$insertion_after_statement,
  repair_expression=paste(deparse(clone$repair_expression),collapse="\n"),
  producer_scripts=lapply(file.path(repair_dir,c("clone_singleton.R","prepare_candidate.R")),identity),
  endpoint="P2_MORPH",batch=302L,cursor=77057L,total=77057L,cell_id=colnames(singleton),
  batch_size_protocol=256L,actual_tail_size=1L,padding_or_dropped_cells=0L,
  full_qc_normalization_cells=ncol(sce),full_normalization_genes=nrow(sce),classifier_genes=nrow(expr),
  normalized_singleton_sha256=objsha(singleton),prediction_parameters=p$prediction,
  input_identities=inputs_before,parent_batch=prior[[301L]],
  rng_before_sha256=objsha(batch301$rng_state),rng_after_sha256=objsha(single$rng),
  dependencies_digest=runtime$dependencies_digest,
  tests=list(real_256_batch_original_matches_cached_prediction_exact=TRUE,
    real_256_batch_original_matches_cached_rng_exact=TRUE,
    real_256_batch_clone_matches_original_prediction_rng_warnings_exact=TRUE,
    original_singleton_failure_reproduced=conditionMessage(failed$value),
    singleton_clone_matches_original_failure_rng_and_warnings_exact=TRUE,
    singleton_matches_two_identical_column_original_oracle_labels_hierarchy_exact=TRUE,
    full_qc_size_factors_csv_sha256_identical=TRUE,namespace_unchanged=TRUE,
    prior_301_batches_all_identity_and_hash_verified=TRUE))
candidate <- list(ids=ids[chunks[[302L]]],prediction=single$value,
                  rng_state=single$rng,dependencies_digest=runtime$dependencies_digest,
                  repair_provenance=provenance)
candidate_path <- file.path(out,"batch_0302.rds")
saveRDS(candidate,candidate_path)
stopifnot(identical(readRDS(candidate_path),candidate))
# Prove no immutable input or prior batch was changed during this process.
stopifnot(identical(inputs_before,lapply(bound,identity)))
for (v in prior) stopifnot(identical(v$rds,identity(v$rds$path)),
                         identical(v$receipt,identity(v$receipt$path)))
stopifnot(!file.exists(file.path(chunkdir,"batch_0302.rds")))
jsonlite::write_json(prior,file.path(out,"prior_batches_verified.json"),auto_unbox=TRUE,pretty=TRUE)
jsonlite::write_json(list(status="passed_candidate_only_not_published",candidate=identity(candidate_path),
  provenance=provenance, prior_batches_manifest=identity(file.path(out,"prior_batches_verified.json")),
  immutable_inputs_and_prior_batches_unchanged=TRUE,production_batch_0302_absent=TRUE,
  warnings=list(multi_original_count=length(multi_original$warnings),
                singleton_count=length(single$warnings),
                singleton_messages=unique(single$warnings))),
  file.path(out,"validation.json"),auto_unbox=TRUE,pretty=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_sessionInfo.txt"))
log_event("PASS: candidate batch 302 saved only under repairs/singleton/candidate")
