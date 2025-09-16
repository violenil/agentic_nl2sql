import json
import glob

DB = "academic"

with open(f"{DB}.json","r") as f:
    data = json.load(f)

f = open(f"questions_{DB}.txt","w")
g = open(f"gold_{DB}.txt","w")

for d in data:
    pergunta = ""
    for s in d['sentences']:
        pergunta = s['text']
        sorted_dict = [aaa[0] for aaa in sorted(s['variables'].items(), key=lambda item: len(item[0]), reverse=True)]
        print(sorted_dict)
        print(pergunta)
        gold = d['sql'][0]
        for v in sorted_dict:
            print(v)
            pergunta = pergunta.replace(v,s['variables'][v])
            gold = gold.replace(v,s['variables'][v])

        f.write(f"\'{pergunta}\'\n")
        g.write(f"{gold}\t{DB}\n")

f.close()
g.close()

dict_csv = {}
for g in glob.glob(f"*.csv"):
    print(g)
    dict_csv[g.split("/")[-1].strip().split(".")[0].strip()] = g.strip()

with open(f"tables_csv_{DB}.json","w") as tables:
    json.dump(dict_csv, tables)


