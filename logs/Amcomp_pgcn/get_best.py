file_list = []
for i in range(1,11):
    file = "Run"+str(i)+".txt"
    file_list.append(file)

with open("best.csv","w") as b:
    for fname in file_list:
        with open(fname) as f:
            lnum = 1
            line = f.readlines()[-lnum]
            while True:
                if (len(line))==0:
                    lnum+=1
                    line = f.readlines()[-lnum]
                else:
                    break
            acc = line[13:20]
            ep = line[30:35]
            b.write("\n"+str(acc)+","+str(ep))