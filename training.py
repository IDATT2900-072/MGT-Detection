import pandas as pd
import csv

data = pd.read_csv("../../Dataset/GPT-wiki-intro.csv", engine="python")
real = data['wiki_intro']
fake = data['generated_intro']

fields = ['label', 'text']
rows = []

for i in range(len(data)):
    rows.append([0, real[i]])
    rows.append([1, fake[i]])

with open('wiki-labeled', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)

print(real[3])
print(fake[3])





