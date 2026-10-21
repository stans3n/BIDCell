# Isolated scClassify 1.5.1 shape repair. Never changes its package namespace.
clone_singleton_predictor <- function() {
  ns <- asNamespace("scClassify")
  original <- get("predict_scClassifySingle", ns)
  target <- quote(predMat <- vapply(seq_len(ncol(predMat)), function(x)
    getPredRes(predMat, cutree_list, x), character(nrow(predMat))))
  statements <- as.list(body(original))
  at <- which(vapply(statements, identical, logical(1), target))
  stopifnot(length(at) == 1L)
  # vapply simplifies character(1) results to a vector. Restore exactly the
  # row/level axes that the following unchanged apply(..., 1, ...) requires.
  repair <- quote(if (is.null(dim(predMat))) {
    stopifnot(ncol(exprsMat_test) == 1L,
              length(predMat) == length(levelModel))
    predMat <- matrix(predMat, nrow = 1L,
                      dimnames = list(colnames(exprsMat_test), NULL))
  })
  patched <- original
  body(patched) <- as.call(append(statements, list(repair), after = at))
  stopifnot(identical(formals(patched), formals(original)),
            identical(as.list(body(patched))[-(at + 1L)], statements))
  isolated <- new.env(parent = ns)
  environment(patched) <- isolated
  isolated$predict_scClassifySingle <- patched
  public <- get("predict_scClassify", ns)
  environment(public) <- isolated
  list(predict = public, single = patched, original_single = original,
       insertion_after_statement = at, repair_expression = repair)
}
