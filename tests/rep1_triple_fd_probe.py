"""Owned engineering process: verify actual inherited OFDs, never unlock."""
import fcntl,json,os,sys,tempfile
from pathlib import Path
rows=[]
for fd in map(int,os.environ['REP1_TRIPLE_SIMPLE_LOCK_FDS'].split(',')):
    st=os.fstat(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows.append(dict(fd=fd,path=os.readlink(f'/proc/self/fd/{fd}'),device=st.st_dev,inode=st.st_ino,owns_same_OFD=True))
target=Path(sys.argv[1]);fd,tmp=tempfile.mkstemp(prefix='.OFD.',dir=target.parent)
with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid(),fds=rows),stream);stream.flush();os.fsync(stream.fileno())
os.replace(tmp,target)
