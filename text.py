i = 1 
cnt = 0
j=1
while i <= 100:
    for j in range(1,i):
      if i % j == 0:
            cnt = cnt+1
      if cnt == 2:
        print(i)
      i = i+1

            
