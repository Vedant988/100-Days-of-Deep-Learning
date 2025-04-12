yash = [2,6,4,56,26,56]
m = yash[0]
sm = yash[0]
for i in range(len(yash)):
    if(yash[i]>sm and yash[i]<m):
        sm = yash[i]
    if(yash[i]>m):
        sm = m
        m=yash[i]

print(f"m:{m}")
print(sm)