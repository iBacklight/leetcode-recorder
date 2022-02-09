# 																			LeetCode 刷题记录

## Preface

The recommended  learn order:

1.   **Data Structure**: Array(数组), Linked list(链表), graph(图), Hash Table(哈希表), String(字符串), 树(tree), Stack and Queue(栈与队列), Bit(位运算)
2.   **Algorithm**: Double pointer(双指针), Sort(排序), Greedy(贪心思想), Dichotomous search(二分查找), Search(搜索), Partition(分治), Dynamic Planning(动态规划), Math(数学)



## Data Structure 数据结构

### Array(数组)

Not too much to add to this structure. Let's go to see the code.

#### LC .21 moveZeroes (Easy)

```
For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
```

错误记录：

a) 忽略了即使从最开始使用枚举变量，但中途改变数组时，idx依旧会跟着改变的情况。

Note the index of each num would be **modified** if we append or pop a number from the array. **My solution** is based on pop a num and append a 0, and record 0 raised times. Since idx of a value would be decreased as numbers of 0 increased, we  can summarize **the actual idx of a value = origin idx -  number of zeros**

```python
def moveZeroes(self, nums: List[int]) -> None:
    zero_cnt = 0
    for n in range(len(nums)):
      if nums[n-zero_cnt] == 0: # key here
        nums.pop(n-zero_cnt)
        nums.append(0)
        zero_cnt += 1
```

However, this is not efficient enough. Only beats 20% on runtime and 89% on memory. The solution exploits double-pointer, runtime 80% but memory 5%.

```python
n = len(nums)
left = right = 0
while right < n:
    if nums[right] != 0: # if right = 0, then not move
        nums[left], nums[right] = nums[right], nums[left]
        left += 1 # left pointer only moves when right side is not 0
    right += 1 # right pointer always move forward 
```

There is also an interesting solution since the key is bool so only reverses 1 and 0.

```python
nums.sort(key=bool, reverse=True)
```



#### LC. 566 Reshape the Matrix (Easy)

```html
Input:
nums =
[[1,2],
 [3,4]]
r = 1, c = 4

Output:
[[1,2,3,4]]

Explanation:
The row-traversing of nums is [1,2,3,4]. The new reshaped matrix is a 1 * 4 matrix, fill it row by row by using the previous list.
```

错误记录：

a）忽略题干中 return origin mat if the r and c are not reasonable. 请仔细审题

b）忽略了原矩阵中行和列的关系.

My solution. 考虑到实际上是重写reshape函数，所以首先要判断given r&c 是否合理, 即乘积是否与原来相等. 之后在双循环下利用i和j来与原矩阵建立联系（在原矩阵中，当列数耗尽时，遍历下一行）

```python
def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        cr = len(mat)
        cc = len(mat[0])
        reshaped_mat = []
        i = j = 0 # i means original row number, j means col
        if cr*cc != r*c:
            return mat
        else:
            for row in range(r):
                reshaped_mat.append([])
                for col in range(c):
                    if j >= cc:# if j > origin col num
                        j = 0 # start traversing next row
                        i += 1
                    reshaped_mat[row].append(mat[i][j])
                    j += 1
        return reshaped_mat
```

Runtime beats 87% and memory usage beats 62 %. The official solution is shown below. 可取之处在于

1. 用for循环代替np.zeros创建r*c的0矩阵,用于后续填补
2. 找到对应的映射关系，即**行数idx为与列数相除，列数idx为与列数取余**. 这一点在新旧两个矩阵中都适用。


```python
def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(nums), len(nums[0])
        if m * n != r * c:
            return nums
        
        ans = [[0] * c for _ in range(r)] # use this to replace np.zeros
        for x in range(m * n):
            ans[x // c][x % c] = nums[x // n][x % n] # map
        
        return ans
```



#### LC. 485 Max Consecutive Ones (Easy)

```
输入：[1,1,0,1,1,1] (注意：输入为二进制数组)
输出：3
解释：开头的两位和最后的三位都是连续 1 ，所以最大连续 1 的个数是 3.
```

错误记录：

a) 只考虑了起始位置但没有考虑终止位置。若终止位置处为最大连续1则在遍历后仍要执行一次判断。请着重考虑终止与起始的不同。

My solution. 

```python
def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max_num = 0
        temp_num = 0
        for n in nums:
            if n == 1:
                temp_num += 1        
            else:
                if temp_num > max_num:
                    max_num = temp_num
                temp_num = 0
        max_num = max(max_num, temp_num)
        return max_num
```

The solution is similar to the official solution, Time complexity is O(n), space complexity is O(1). An interesting solution in the discussion area:

```python
def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        cur = 0
        maximum = 0
        for num in nums:
            cur = (cur+num)*num # 加上当前数字再乘以当前数字，等同于遇到一加一，遇到零清零
            if cur > maximum:
                maximum = cur 
        return maximum
```



#### LC.240 Search a 2D Matrix II (Medium)

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

* 每行的元素从左到右升序排列。
* 每列的元素从上到下升序排列。

链接：https://leetcode-cn.com/problems/search-a-2d-matrix-ii/solution/sou-suo-er-wei-ju-zhen-ii-by-leetcode-so-9hcx/

本题如果**双循环强解**, 时间复杂度为O(mn). 但如果使用如下**Z字形搜索**，从矩阵右上角开始（记录x,y 分别为行数起始，以及列数最后），即行从右往左遍历（降序），列从上往下遍历（升序）。该方法通用处在于：

* mat(x,y) > traget: 说明mat(x,y)所在y列以下的数字均大于target，故该列不必继续寻找，寻找左边一列. y--
* mat(x,y) < traget: 说明mat(x,y)所在x行往左的数字均小于target，故该行不必继续寻找， 寻找下面一行.x++
* mat(x,y) = traget: return True

```python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        x, y = 0, n - 1
        while x < m and y >= 0:
            if matrix[x][y] == target:
                return True
            if matrix[x][y] > target:
                y -= 1
            else:
                x += 1
        return False
```

这种方法由于x最多增加m次，y最多减少n次，且同时进行，故时间复杂度为O(m+n). 大大缩小了程序运行时间.除此之外， 还有二分法查找，时间复杂度为 O(logn)：

```python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            idx = bisect.bisect_left(row, target) # 
            if idx < len(row) and row[idx] == target:
                return True
        return False
```



#### LC. 378 Kth Smallest Element in a Sorted Matrix ((Medium))

给你一个 n x n 矩阵 matrix ，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
请注意，它是 排序后 的第 k 小元素，而不是第 k 个 不同 的元素。
链接：https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix

输入：matrix = [[1,5,9],[10,11,13],[12,13,15]], k = 8
输出：13
解释：矩阵中的元素为 [1,5,9,10,11,12,13,13,15]，第 8 小元素是 13

My solution. Good to know that I have considered all conditions. However, since not know ```sorted```function, I didn't try violate method in the very first place: **flatten the mat and sort them, and try to pick up kth value**. (can beats 33% on runtim and 8% on memory usage).

```python
return sorted(sum(matrix, []))[k-1]
```

Instead, **I traverse every value and store k-1 minimum values in a list. Dynamically modify the values in the list and choose the maximum value in the list as the lower bound of kth value, done**. It only beats 5% on runtime and 20% on memory usage among all leetcoders.

```python
 def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        kth_num = 1e9+1
        last_kth_num = -inf
        flatten_kminus1_mat = []
        cnt = 0
        for r in range(n):
            for c in range(n):
                cnt += 1 
                if cnt < k:
                    flatten_kminus1_mat.append(matrix[r][c])
                    if cnt == k-1:
                        last_kth_num = max(flatten_kminus1_mat)
                        last_index = flatten_kminus1_mat.index(last_kth_num)
                        if isinstance(last_index,list) == True:
                            last_index = last_index[0]
                else:
                    if matrix[r][c] < kth_num and matrix[r][c] >= last_kth_num:
                        kth_num = matrix[r][c]
                    elif matrix[r][c] < kth_num and matrix[r][c] < last_kth_num:
                        flatten_kminus1_mat.pop(last_index)
                        flatten_kminus1_mat.append(matrix[r][c])
                        kth_num = last_kth_num
                        last_kth_num = max(flatten_kminus1_mat)
                        last_index = flatten_kminus1_mat.index(last_kth_num)
                        if isinstance(last_index,list) == True:
                            last_index = last_index[0]
                    else:
                        # rest in this col would be greater than current one, so no need to search
                        break
        return kth_num
```

官方给出的答案非常tricky加fancy. 总的来说是利用**二分法**作为基础，从左下角开始沿着二分边缘向上检索. 

* 检索值若小于mid（最大值+最小值/2， 分别为左上角和右小角值），则将当前所在列的不大于 mid 的数的数量（即 i + 1）累加到答案中，并向右移动，否则向上移动；

* 不断移动直到走出格子为止

  每次对于「猜测」的答案 mid，计算矩阵中有多少数不大于 mid ：


* 如果数量不少于 k，那么说明最终答案 x 不大于 mid；
* 如果数量少于 k，那么说明最终答案 x 大于 mid。

![fig3](https://assets.leetcode-cn.com/solution-static/378/378_fig3.png)

代码如下：

```python
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        def check(mid):# 二分边缘查找
            i, j = n - 1, 0
            num = 0
            while i >= 0 and j < n:
                if matrix[i][j] <= mid:
                    num += i + 1
                    j += 1
                else:
                    i -= 1
            return num >= k

        left, right = matrix[0][0], matrix[-1][-1]
        while left < right:
            mid = (left + right) // 2 #计算mid
            if check(mid):
                right = mid # 二分检索数目大于等于k，说明答案在mid二分边缘的左半边（小于等于mid）
            else:
                left = mid + 1 # 二分检索数目小于k，说明答案在mid二分边缘的右半边（大于mid）
                
    	return left
```



#### LC. 645 Set Mismatch (Easy)

集合 s 包含从 1 到 n 的整数。不幸的是，因为数据错误，导致集合里面某一个数字复制了成了集合里面的另外一个数字的值，导致集合 丢失了一个数字 并且 有一个数字重复 。

给定一个数组 nums 代表了集合 S 发生错误后的结果。

请你找出重复出现的整数，再找到丢失的整数，将它们以数组的形式返回。
链接：https://leetcode-cn.com/problems/set-mismatch

```
输入：nums = [1,2,2,4]
输出：[2,3]
```

```
输入：nums = [1,1]
输出：[1,2]
```

本题要点： 

a. 数组本身是**无序**的 

b. **重复的数字不一定相邻**，缺失的字符一定在n之中

由于本题解法过多(暴力循环判断解法， HashTable, 数学方法(python的set用法))，故不多列举解法。本体中可以学到使用collections模组中的counter以及其对应的most_common用法.

```python
def findErrorNums(self, nums: List[int]) -> List[int]:
    from collections import Counter
    check = Counter(nums)
    dup = check.most_common(1)[0][0]
    for i in range(1, len(nums) + 1):
        if check[i] == 0:
            abs = i
            break
            # abs = [i for i in range(1, len(nums) + 1) if check[i] == 0][0]
            return [dup, abs]
```

Counter 用法可以参考最后一节 **Useful algorithms&elements in Array 有用的算法和玩意儿**. 



#### LC. 287 Find the Duplicate Number (Medium)

给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设 nums 只有 一个重复的整数 ，找出 这个重复的数 。你设计的解决方案必须不修改数组 nums 且只用常量级 O(1) 的额外空间**（故本题不可以使用哈希表）**。

```
输入：nums = [1,3,4,2,2]
输出：2
```

链接：https://leetcode-cn.com/problems/find-the-duplicate-number

错误记录:

* 第一次提交使用双循环，但是执行时间超时。复杂度O(n^2)

解决本题用到了dict().keys()来判断此数字是否遍历过，从而判断该数字是否重复

```python
def findDuplicate(self, nums: List[int]) -> int:
        cnt = dict()
        for num in nums:
            if num in cnt.keys():
                return num
            else:
                cnt[num] = 1
```

或者参考645题中的数学算法(重复的总和除以重复的次数)

```python
return (sum(nums) - sum(set(nums))) // (len(nums) - len(set(nums)))
```

而官方给出的两种方法则都非常有趣和魔幻。第一种是二分搜索，这道题告诉我们，如果不在作用于数组本身，而作用于数组中的（潜在有序的，比如从1到n都有）元素，则即使数组本身是无序的也可以使用二分法。

二分查找的思路是先猜一个数（有效范围 [left, right] 里位于中间的数 mid），然后统计原始数组中 小于等于 mid 的元素的个数 cnt：

* 如果 cnt **严格大于** mid。根据抽屉原理，重复元素就在区间 [left,mid] 里；
* 否则，重复元素就在区间 [mid + 1, right] 里。

参考代码：

```python
 # Bisect Search
        left, right = 1, len(nums)-1
        cnt = 0
        while(left <= right):
            mid = (left+right)//2
            for num in nums: # 每一次更换mid 均要全部遍历数组一次
                if  num <= mid:
                    cnt += 1 # 统计数组中不大于mid的数目
            if cnt <= mid:# 此时说明区间大于mid,“<” 可能发生在重复了一次以上的情况
                left = mid + 1
            else:#严格大于mid
                right = mid-1
                dup = mid
            cnt = 0
        return dup
```


第二种是**快慢指针**，双指针的一种应用。快慢指针由于比较复杂，请参考这篇解答。答主用非常直接的图片形式给出了如何使用双（快慢）指针以及为什么会回到循环的起始点。链接：https://leetcode-cn.com/problems/find-the-duplicate-number/solution/287xun-zhao-zhong-fu-shu-by-kirsche/

代码参考:

```python
# Fast-slow pointer
        slow = fast = cir_start = 0
        while True:
            fast = nums[nums[fast]]
            slow = nums[slow]
            if fast == slow: # utilize fast pointer find loop
                break

        while True: 
            slow = nums[slow]
            cir_start = nums[cir_start]# set same pace for two pointer to find dup
            if cir_start == slow:
                return slow        
```



#### LC. 697. Degree of an Array (Easy)

给定一个非空且只包含非负数的整数数组 nums，数组的 度 的定义是指数组里任一元素出现频数的最大值。

你的任务是在 nums 中找到与 nums 拥有相同大小的度的最短连续子数组，返回其长度。

输入：nums = [1,2,2,3,1,4,2]
输出：6
解释：
数组的度是 3 ，因为元素 2 重复出现 3 次。
所以 [2,2,3,1,4,2] 是最短子数组，因此返回 6 。

错误记录：

1> 第一次测试未考虑多个数字拥有相同的degree但其对应的连续数组长度不同。

My solution is similar to the official solution. **Hash Table record degree/first idx/last idx.**

```python
def findShortestSubArray(self, nums: List[int]) -> int:
        hash = dict()
        max_d = 0
        max_num = nums[0]
        for idx, num in enumerate(nums):
            if num in hash.keys():
                hash[num]['cnt'] += 1
                hash[num]['last_idx'] = idx
            else:
                hash[num] = {'cnt':1, 'last_idx':idx, 'first_idx':idx}
            if  hash[num]['cnt'] > max_d:
                max_d =  hash[num]['cnt']
                max_num = num
            elif hash[num]['cnt'] == max_d:
                 if (hash[num]['last_idx'] - hash[num]['first_idx']) < (hash[max_num]												['last_idx'] - hash[max_num]['first_idx']):
                     max_num = num
            else:
                pass
        
        return (hash[max_num]['last_idx'] - hash[max_num]['first_idx'] + 1)
```





#### Useful algorithms&elements in Array 有用的算法和玩意儿

1. ##### 二分查找 Bisect search

   要求**数组必须有序**，执行一次的时间复杂度为O(logn)

- 从数组的中间元素开始，如果中间元素正好是要查找的元素，则搜素过程结束；
- 如果某一特定元素大于或者小于中间元素，则在数组大于或小于中间元素的那一半中查找，而且跟开始一样从中间元素开始比较。
- 如果在某一步骤数组为空，则代表找不到。

```python
def binary_search_loop(lst,value):
    low, high = 0, len(lst)-1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] < value:
            low = mid + 1
        elif lst[mid] > value:
            high = mid - 1
        else:
            return mid
    return None
```

Python 有一个 `bisect` 模块，用于维护有序列表。`bisect` 模块实现了一个算法用于插入元素到有序列表。在一些情况下，这比反复排序列表或构造一个大的列表再排序的效率更高。Bisect 是二分法的意思，这里使用二分法来排序，它会将一个元素插入到一个有序列表的合适位置，这使得不需要每次调用 sort 的方式维护有序列表。模块具体内容请参考：https://docs.python.org/zh-cn/3.6/library/bisect.html

本分享参考来源：http://kuanghy.github.io/2016/06/14/python-bisect

------------------------

总结一下Leetcode题中需要考虑二分法的触发条件：

* 可能的返回值区间是连续的（可以是空间连续，也可以是数值连续）。比如是从[1,2,...,n]中的某个数，或比如是一个连续数组的某个元素。
* 能够通过一个特定返回值的某种性质，判断真正返回值的区间。
* 每次检查的函数时间复杂度应该小于等于O(n)。如果检查函数太复杂，在lc的题库里，就算写成二分应该也是通过不了的。

作者：accsrd
链接：https://leetcode-cn.com/problems/find-the-duplicate-number/solution/python3-wei-shi-yao-wo-men-yao-yong-er-f-0y0x/

---------

2. ##### Python Counter

```python
from collections import Counter
```

Counter 的本质就是一个特殊的 dict，只不过它的 key 都是其所包含的元素，而它的 value 则记录了该 key 出现的次数。因此，**如果通过 Counter 并不存在的 key 访问 value，将会输出 0**（代表该 key 出现了 0 次）。 

程序可通过任何可法代对象参数来创建 Counter 对象，此时 Counter 将会自动统计各元素出现的次数，并以元素为 key，出现的次数为 value 来构建 Counter 对象；程序也能以 dict 为参数来构建 Counter 对象；还能通过关键字参数来构建 Counter 对象。

```python
c3 = Counter(['Python', 'Swift', 'Swift', 'Python', 'Kotlin', 'Python'])
# Output: Counter({'Python': 3, 'Swift': 2, 'Kotlin': 1})
```

Counter 继承了 dict 类，因此它完全可以调用 dict 所支持的方法。此外，Counter 还提供了如下三个常用的方法：

- elements()：该方法返回该 Counter 所包含的**全部元素组成的迭代器**。
- most_common([n])：该方法返回 Counter 中**出现最多的 n 个元素**。
- subtract([iterable-or-mapping])：该方法计算 Counter 的减法，**其实就是计算减去之后各元素出现的次数**。 下面程序示范了 Counter 类中这些方法的用法示例：

```python
chr_cnt = Counter('abracadabra')
print(chr_cnt.most_common(3))  
# Output: [('a', 5), ('b', 2), ('r', 2)]
```

```python
c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
c.subtract(d)
print(c) 
# Output: Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})
```

参考：https://www.ixyread.com/read/ID1605494154VwJr/OEBPS-Text-Section0121.html





##  Linked list(链表)

A linked table is either **empty nodes** or has a value and a pointer to the next linked table, so many linked table problems can be handled with **recursion**.

Regarding Recursion: 1) Separate a problem into several sub-problems 2) Sub-problems share exact same solution ideas. and 3) There should be a baseline or terminated condition.



