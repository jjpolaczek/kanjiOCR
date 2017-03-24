from __future__ import print_function  # Only needed for Python 2

with open("mapping") as f:
    content = f.readlines()
with open("mapping.py", 'w') as f:
    line = "def JISX201Dict():\n    d = dict()\n"
    f.write(line.encode('utf8'))
    for l in content:
        sline = l.split()
        line = (u"    d[%d] = '%s'\n" % \
               (int(sline[0],16), unichr(int(sline[1],16))))
        f.write(line.encode('utf8'))
    line = "    return d"
    f.write(line.encode('utf8'))

with open("jisx0208") as f:
    content = f.readlines()
with open("mapping0208.py", 'w') as f:
    line = "def JISX0208Dict():\n    d = dict()\n"
    f.write(line.encode('utf8'))
    for l in content:
        sline = l.split()
        line = (u"    d[%d] = '%s'\n" % \
               (long(sline[1],16), unichr(long(sline[2],16))))
        f.write(line.encode('utf8'))
    line = "    return d"
    f.write(line.encode('utf8'))
