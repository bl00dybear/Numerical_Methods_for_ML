n = int(input())

x=0.1

for i in range (1,n+1,1):
    print(f"pas{i} {x:.4f}")
    if x<0.5:
        x*=2
    else:
        x*=2
        x-=1

print(x)