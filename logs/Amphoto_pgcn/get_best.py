file_list = []
for i in range(1,14):
    file = "Run"+str(i)+".txt"
    file_list.append(file)

with open("best.csv","w") as b:
    for fname in file_list:
        with open(fname) as f:
            lnum = 1
            lines = f.read().splitlines()
            while (len(lines[lnum*-1]))==0:
                lnum+=1
            line = lines[lnum*-1]
            acc = line[13:20]
            ep = line[30:35]
            b.write("\n"+str(acc)+","+str(ep))