# Engineering source-resolution witness; never a scientific evaluator.
args <- commandArgs(trailingOnly=TRUE)
writeLines("audited_snapshot_R_original",file.path(args[1],"R_value.txt"))
writeLines(normalizePath(args[2],mustWork=TRUE),file.path(args[1],"R_source.txt"))
writeLines("succeeded",file.path(args[1],"execution_status.txt"))
