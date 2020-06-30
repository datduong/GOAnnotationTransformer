import csv
import sys

rows = []
with open(sys.argv[1]) as csv_file:
   csv_reader = csv.reader(csv_file,delimiter='\t')
   for row in csv_reader:
      r = []
      r.append(row[0])
      r.append(row[1].replace(" ", ""))
      goes = row[2].split()
      gostr = set()
      for i in range(0, len(goes)):
        go = goes[i][0:2] + ":" + goes[i][2:]
        gostr.add(go)

      r.append(gostr)
      rows.append(r)

import pandas as pd

df = pd.DataFrame(rows, columns=["proteins", "sequences", "annotations"])
df.to_pickle(sys.argv[2])
