def sort_arr(a, k):
    print('Here are the outputs:')
    a.sort(key=lambda x: (x[k + 1], x[0]))
    for i in range(len(a)):
        for j in range(1, len(a[i]) - 1):
            print(a[i][j], end=' ')
        print(a[i][-1])
    
print('Please provide inputs:')
n, m = map(int, input().split())
arr = []
for i in range(n):
    cur_line = [i] + list(map(int, input().split()))  
    arr.append(cur_line)
k = int(input())

sort_arr(arr, k)
