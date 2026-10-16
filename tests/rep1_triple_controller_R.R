# Isolated engineering fixture; no biological production or formal admission.
args<-commandArgs(trailingOnly=TRUE);out<-args[1];mode<-args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
fds<-as.integer(strsplit(Sys.getenv('REP1_TRIPLE_SIMPLE_LOCK_FDS'),',',fixed=TRUE)[[1]])
describe<-function()list(pid=Sys.getpid(),fds=lapply(fds,function(fd)list(fd=fd,path=Sys.readlink(paste0('/proc/self/fd/',fd)))))
jsonlite::write_json(describe(),file.path(out,'R_identity.json'),auto_unbox=TRUE)
if(mode=='fail')stop('intentional real R nonzero engineering fixture')
if(mode=='orphan') {
 child<-parallel::mcparallel({
   jsonlite::write_json(describe(),file.path(out,'R_grandchild.json'),auto_unbox=TRUE)
   status<-system2('/home/zwan0726/miniforge3/envs/bidcell_local/bin/python',
     c('-I','-S','-B',shQuote(file.path(getwd(),'tests/rep1_triple_fd_probe.py')),shQuote(file.path(out,'R_grandchild_OFD.json'))))
   stopifnot(status==0L);Sys.sleep(6);writeLines('succeeded',file.path(out,'grandchild_done.txt'));TRUE
 },silent=TRUE)
 parallel::mccollect(child)
} else {
 source<-normalizePath(file.path(getwd(),'external_sources/publish_utils.R'),mustWork=TRUE)
 stopifnot(startsWith(source,paste0(getwd(),'/')))
 base::source(source,local=.GlobalEnv)
 jsonlite::write_json(list(path=source,sha256=digest::digest(source,algo='sha256',file=TRUE,serialize=FALSE)),file.path(out,'R_source.json'),auto_unbox=TRUE)
}
writeLines('succeeded',file.path(out,'execution_status.txt'))
