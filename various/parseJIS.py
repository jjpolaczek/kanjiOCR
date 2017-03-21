from __future__ import print_function  # Only needed for Python 2

with open("mapping") as f:
    content = f.readlines()
with open("mapping.py", 'w') as f:
    for l in content:
        sline = l.split()
        line = (u"d[%d] = '%s'\n" % \
               (int(sline[0],16), unichr(int(sline[1],16))))
        f.write(line.encode('utf8'))
