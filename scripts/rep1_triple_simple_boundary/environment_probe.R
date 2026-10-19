# Read-only package identity under the exact formal R environment.
p <- installed.packages()
rows <- lapply(order(p[,"Package"],p[,"LibPath"],method="radix"),function(i)
 list(package=p[i,"Package"],version=p[i,"Version"],library=p[i,"LibPath"]))
stopifnot(as.character(packageVersion("entropy"))=="1.3.2")
cat(jsonlite::toJSON(list(version=R.version.string,home=R.home(),libraries=.libPaths(),
 entropy=as.character(packageVersion("entropy")),packages=rows),auto_unbox=TRUE))
