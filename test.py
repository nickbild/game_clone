
vals = []
for line in open("test.txt"):
    vals = line.strip().split(" ")
    

for y in range(120):
    for x in range(160):
        r = int(vals[x+(160*y)])

        # if r == 0:
        #     r = 0
        # elif r == 1:
        #     r = 130
        # elif r == 2:
        #     r = 234
        # else:
        #     print("ERROR")
        print(str(r) + " ", end="")

    print("")
