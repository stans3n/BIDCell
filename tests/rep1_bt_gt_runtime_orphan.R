# Engineering-only real R child: hold inherited descriptors while Python dies.
args <- commandArgs(trailingOnly=TRUE)
writeLines(as.character(Sys.getpid()),file.path(args[1],"R.pid"))
Sys.sleep(7)
writeLines("succeeded",file.path(args[1],"execution_status.txt"))
