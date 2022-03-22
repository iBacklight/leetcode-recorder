# 																			LeetCode 刷题记录

## Preface

The recommended  learn order:

1.   **Data Structure**: Array(数组), Linked list(链表), graph(图), Hash Table(哈希表), String(字符串), 树(tree), Stack and Queue(栈与队列), Bit(位运算)
2.   **Algorithm**: Double pointer(双指针), Sort(排序), Greedy(贪心思想), Dichotomous search(二分查找), Search(搜索), Partition(分治), Dynamic Planning(动态规划), Math(数学)



## Data Structure 数据结构

此节包括 Array(数组), Linked list(链表), graph(图), Hash Table(哈希表), String(字符串), 树(tree), Stack and Queue(栈与队列), Bit(位运算)

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



#### LC.240 Search a 2D Matrix II (Medium)*

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



#### LC. 378 Kth Smallest Element in a Sorted Matrix ((Medium))*

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



#### LC. 645 Set Mismatch (Easy)*

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



#### LC. 287 Find the Duplicate Number (Medium)*

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



#### LC. 697 Degree of an Array (Easy)

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



#### LC. 766 Toeplitz Matrix (Easy)

给你一个 m x n 的矩阵 matrix 。如果这个矩阵是托普利茨矩阵，返回 true ；否则，返回 false 。

如果矩阵上每一条由左上到右下的对角线上的元素都相同，那么这个矩阵是 托普利茨矩阵 。
链接：https://leetcode-cn.com/problems/toeplitz-matrix

```
输入：matrix = [[1,2,3,4],
			   [5,1,2,3],
			   [9,5,1,2]]
输出：true

解释：
在上述矩阵中, 其对角线为: 
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]"。 
各条对角线上的所有元素均相同, 因此答案是 True 。
```

这道题比较简单，遍历每一个元素以及其对角元素即可。解答与答案一致。解答中有一个有意思的答案，**左斜看和rc_sum，右斜看差rc_diff**。

```python
def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        R, C = len(matrix), len(matrix[0])
        rc_diff = defaultdict(int)          #八皇后问题的热身
        for r in range(R):
            for c in range(C):
                if r-c not in rc_diff:
                    rc_diff[r-c] = matrix[r][c] #不清楚这有什么数学原理吗？
                else:
                    if rc_diff[r-c] != matrix[r][c]:
                        return False
        return True

作者：code_learner
链接：https://leetcode-cn.com/problems/toeplitz-matrix/solution/c-python3-ba-huang-hou-wen-ti-de-re-shen-ppu5/
```



#### LC. 565 Array Nesting (Medium)

索引从0开始长度为N的数组A，包含0到N - 1的所有整数。找到最大的集合S并返回其大小，其中 S[i] = {A[i], A[A[i]], A[A[A[i]]], ... }且遵守以下的规则。

假设选择索引为i的元素A[i]为S的第一个元素，S的下一个元素应该是A[A[i]]，之后是A[A[A[i]]]... 以此类推，不断添加直到S出现重复的元素。
链接：https://leetcode-cn.com/problems/array-nesting

```
输入: A = [5,4,0,3,1,6,2]
输出: 4
解释: 
A[0] = 5, A[1] = 4, A[2] = 0, A[3] = 3, A[4] = 1, A[5] = 6, A[6] = 2.

其中一种最长的 S[K]:
S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}
```

错误总结：

a. 考虑到快慢指针，但是没有考虑冗余（重复）计算。

b. 试图记录过程中的值存在hash table中，但是记录方式有问题。没有考虑到像解答二一样设置一个暂时值

My solution。记录遍历过的idx并且储存在hash table中。但这样计算占用时间过长。Only beats 25% both on runtime and memory. 基本上也是答案的解。

```python
def arrayNesting(self, nums: List[int]) -> int:
        # fast-slow pointer
        counter = max_num= 0
        hash = dict()
        for slow in range(len(nums)):
            counter = 0
            fast = slow
            if slow in hash.keys():
                continue
            else:
                hash[slow] = 0
            while True:
                fast = nums[nums[fast]]
                slow = nums[slow]
                counter += 1
                 if fast == slow :
                    hash[temp] = counter
                    break
                 elif slow in hash.keys():
                    counter += hash[slow]
                    break
                 else:
                    hash[slow] = 0
            if  counter > max_num:
                max_num = counter
        return max_num
```

另外一个更快更好的, 将遍历过的指针的从那天number也记录在hash table中，则如果在其他数为起始值的环中途遇到该数，则执行：**该起始值对应的哈希表记录应当为=当前cnt + 遇到数的hash table值**。

```python
 def arrayNesting(self, nums: List[int]) -> int:
 		dic = {}
        l = len(nums)
        tmp = [1 for _ in range(l)]
        for i in range(l):
            if tmp[i]:
                j = i
                ct = 0
                while tmp[j]:
                    tmp[j] = 0
                    ct += 1
                    j = nums[j]
                if j in dic:
                    dic[i] = dic[j] + ct # key
                else:
                    dic[i] = ct
        return max(dic.values())
```



#### LC. 769 Max Chunks To Make Sorted (Medium)

给定一个长度为 n 的整数数组 arr ，它表示在 [0, n - 1] 范围内的整数的排列。

我们将 arr 分割成若干 块 (即分区)，并对每个块单独排序。将它们连接起来后，使得连接的结果和按升序排序后的原数组相同。

返回数组能分成的最多块数量。
链接：https://leetcode-cn.com/problems/max-chunks-to-make-sorted

```
输入: arr = [1,0,2,3,4]
输出: 4
解释:
我们可以把它分成两块，例如 [1, 0], [2, 3, 4]。
然而，分成 [1, 0], [2], [3], [4] 可以得到最多的块数
```

错误记录：

* 想到了arr[idx]与idx的关系，却未将idx与当前的max值的关系总结出来。总结归纳类题目仍需大量练习。

Solution

```python
def maxChunksToSorted(self, arr: List[int]) -> int:
        cnt = 0
        cur_max = -1
        n = len(arr)
        for idx in range(n):       
            cur_max = max(cur_max, arr[idx])
            if cur_max == idx:
                cnt += 1
        return cnt
```

The key of this problem is 

```python
if cur_max == idx:
```

Since we only have the max num n-1 and n nums, it is obvious that the index of each num has a strong relationship with their position. For such sub-chunk, **the max number of a chunk should be exactly equal to the index of the max num**. and only in that way it satisfies:

* The chunk includes all of the (k+1) num smaller than max num arr[k]: k, since it has k num with the max num == k (pre-requisite: no repeated nums + start from 0);
* The following n-k nums will not affect the chunk;
* The next chunk starts from k+1, the logic would be same with above rules.



#### Useful algorithms&elements in Array 有用的算法和玩意儿

##### 二分查找 Bisect search

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

##### Python Counter

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

---

3. 哈希表 (Hash table)

   https://www.cnblogs.com/czboke/p/13572409.html

   https://codingdict.com/article/4843

4. 双指针(Double pointer)

----



##  Linked list(链表)

A linked table is either **empty nodes** or has a value and a pointer to the next linked table, so many linked table problems can be handled with **Recursion**. **A single linked list node** includes two different parts:

a) Value: Could be strings, integers, Objects

b) The next node

In this way, we could define the single linked list as a node class:

```python
class linkedlistNode():
	def __init__(self, value, next=None):
		self.value = value
		self.next = next
        
    def get_next(self):
        return self.next
    
    def set_next(self. node):
        self.next = node
        
    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value
```

And thus  lionked list class only:

```python
class linkedlist():
	def __init__(self, head=None):
		self.root = head
        self.size = 0
     
    def get_size(self):
        return self.size
    
    def add(self,value):
        new_node = linkedlistNode(value, self.root)
        self.root = new_node
        self.size += 1
        
    def remove(self, value):
        cur_node = self.root
        prev_node = None
        while cur_node:
            if cur_node.get_value == value:
                if prev_node:
                    prev_node.set_next(cur_node.get_next())
                 else:
                    self.root = cur_code.get_next()
                 self.size -= 1
                 return True
             else:
                prev_node = cur_node
                cur_node = cur_node.get_next()
     
    def find(self, value):
        cur_node  =  self.root
        while cur_node:
            if cur_node.get_data() == value:
                return cur_node
            else:
                cur_node = cur_node.get_next()
        return None  
```



Type of linked list:

![image-20220217175814829](C:\Users\Alex Qi\AppData\Roaming\Typora\typora-user-images\image-20220217175814829.png)

Here is the basic function of a linked list class:![image-20220217175906390](C:\Users\Alex Qi\AppData\Roaming\Typora\typora-user-images\image-20220217175906390.png)

编辑链表有关代码时，务必注意的几个问题

* 注意在python中编辑链表，由于python本身模糊了指针的意向，故实际上链表被旭化成一个字典，而字典本体就是链表指针。若两个链表中的某个节点拥有相同的值，但节点指针不一样，那它们就不属于同一个链表。这一点在160题目中表达的很准确。

* 当遍历节点想要依次改变链表的顺序或者取值，注意生成新链表时，切记生成两次

  ```python
  new,cur = ListNode, ListNode
  # or
  new = ListNode()
  cur = new_node
  ```

  这两个的指针在初始节点是一样的，但后续可以将cur的指向不停更改（此时可能还需借助一个temp空间暂时记录cur或者cur.next的值），而new一直指向链表初始位置，最终new将记录cur的所有变化形成最终需要的链表。详见2题，206题。

---

Regarding **Recursion**: 

1) Separate a problem into several sub-problems 

2) Sub-problems share exact same solution ideas. and 

3) There should be a baseline or terminated condition.

   

#### LC. 2 Two Add (Easy)

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
链接：https://leetcode-cn.com/problems/add-two-numbers

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

My solution. 传统强解，分别计算num1和num2后再进行相加，然后分链表

```python
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        num1 = 0
        num2 = 0
        cur_node = l1
        i = 0
        while cur_node:
             num1 += cur_node.val*pow(10,i)
             cur_node = cur_node.next
             i += 1
        cur_node = l2
        i = 0
        while cur_node:
             num2 += cur_node.val*pow(10,i)
             cur_node = cur_node.next
             i += 1
        new_num = num1 + num2
        new_node = ListNode()
        cur_node = new_node
        while True:
            cur_node.val = new_num % 10
            new_num = new_num//10
            if new_num != 0:
                cur_node.next = ListNode()
                cur_node = cur_node.next
            else:
                break
        return new_node
```

更好的解答如下，利用单链表和加法原理，实现一次循环解决问题。其要点在于优先更新next的值后，再补充更新当前的值；在循环语句最后增加的`tp.val > 9`保证了最后一次运算是二位数的例外情况。此算法增强了运算时间（一次循环）以及减少了内存消耗。

```python
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
	newPoint = ListNode(l1.val + l2.val)
    rt, tp = newPoint, newPoint
    while (l1 and (l1.next != None)) or (l2 and (l2.next != None)) or (tp.val > 9):
        l1, l2 = l1.next if l1 else l1, l2.next if l2 else l2
        tmpsum = (l1.val if l1 else 0) + (l2.val if l2 else 0)
        tp.next = ListNode(tp.val//10 + tmpsum)
        tp.val %= 10
        tp = tp.next
        return rt
```

![image-20220217211109310](C:\Users\Alex Qi\AppData\Roaming\Typora\typora-user-images\image-20220217211109310.png)



#### LC. 160 Intersection of Two Linked Lists (Easy)

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 None 。

链接：https://leetcode-cn.com/problems/intersection-of-two-linked-lists/

![img](https://assets.leetcode.com/uploads/2021/03/05/160_example_1_1.png)



```
输入：listA = [4,1,8,4,5], listB = [5,6,1,8,4,5]
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
```

错误记录：

* 注意题目中是找出相交的链表节点（地址），而不是值。如示例，虽然AB链表均有1，但由于不在同一个指针下，所以不是相交节点。
* 循环判断的结束位置，需要将两个链表都遍历结束，而不应该当结束在其中一个链表遍历结束。

My solution. 利用哈希表，这种解法其实有两种。一是两个链表同时遍历，这样的事件复杂度可能是O(max(m,n))==O(m+n),但是空间复杂度稍大O(m+n)；或者先遍历一个链表, 存入哈希表，然后再遍历另外一个链表。

Hash table 1:

```python
 def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # Hash table
        curA = headA
        curB = headB
        hash_midA = dict()
        hash_midB = dict()
        while (curA != None ) or (curB != None):
            if curA:
                hash_midA[curA] = 0
                if curA in hash_midB.keys():
                    return curA
                curA = curA.next
            if curB:
                hash_midB[curB] = 0 
                if curB in hash_midA.keys():
                    return curB
                curB = curB.next 
        return None
```

Hash table 2:

```python
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        s = dict()
        while headA:
            s[headA] = 0
            headA = headA.next
        while headB:
            if headB in s.keys():
                return headB
            headB = headB.next
        return None
```

另外一个进阶算法，利用了双指针(Double Pointer), 两个链表同时开始遍历，当一个链表遍历结束后则将其指向另外一个链表头。要点在于：

* 当两个链表存在相交节点：由于两个链表走过相同相同的路程（到相交节点），又是同时开始遍历，所以一定会在相交节点处相遇。
* 当两个链表不存在交点：最终会在对方的None处相遇。

两个指针均会遍历两个链表的各个节点各一次，时间复杂度O(m+n)。而且无需多余储存空间， 空间复杂度O(1)。

```python
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        a,b = headA,headB
        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a
```

另外也可以求两个链表的差，长的链表先移动，移动到和短的链表一样长的程度，然后两个链表同步往后边移动边比较。

参考第三种解法：https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/acm-xuan-shou-tu-jie-leetcode-xiang-jiao-c8zo/



#### LC. 206 Reversed Linked List (Easy)*

Note: 此题目为链表中的经典解题，需详细理解。

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

![img](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)



本题是链表经典题目，解答涉及哈希，迭代(Iteration), 递归(Recursion)

**a. Hash Table**

思路较为简单，利用哈希表记录节点位置和对应的val的键对值，再反向赋予另外一个链表

```python
def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return None
        l = 0 # Calculate the length of the linked list
        hash_mid = dict()
        while head.next != None:
            hash_mid[l] = head.val
            l += 1
            head = head.next
        new_node = temp = ListNode(hash_mid[l-1])
        for n in reversed(range(l-1)): # re-apply hash table in reversed rule
            temp.next = ListNode(hash_mid[n])
            temp = temp.next
        return new_node 
```

**b. 迭代**

在遍历链表时，将当前节点的 next指针改为指向前一个节点。由于节点没有引用其前一个节点，因此必须事先存储其前一个节点。在更改引用之前，还需要存储后一个节点。最后返回新的头引用。

![迭代.gif](https://pic.leetcode-cn.com/7d8712af4fbb870537607b1dd95d66c248eb178db4319919c32d9304ee85b602-%E8%BF%AD%E4%BB%A3.gif)

Code:

```python
def reverseList(self, head: ListNode) -> ListNode:
        # Simple version of iteration
        if not head:
             return None
        prev = None
        curr = head # head pointer
        while curr != None:
            temp = curr.next # here temp will store the curr.next pointer, and iter
            curr.next = prev
            prev = curr
            curr = temp # give back, for next loop step iter to next node
        return prev
```

**c. 递归**

递归的两个条件：

1. 终止条件是当前节点或者下一个节点==null
2. 在函数内部，改变节点的指向

在本题中，我们可以先节点指针遍历到链表尾部（此时递归前进，一直调用递归函数），之后再反向返回每个递归回合的当前节点指针（递归反向，即开始return）

![递归.gif](https://pic.leetcode-cn.com/dacd1bf55dec5c8b38d0904f26e472e2024fc8bee4ea46e3aa676f340ba1eb9d-%E9%80%92%E5%BD%92.gif)

Code:

```python
def reverseList(self, head: ListNode) -> ListNode:
        # Recursion
        def Recur(head):
            if head == None or head.next == None:# head == None avoids input == []
                # return until traverse to second last node of origin linked list
                return head, head
            pre, last = Recur(head.next)
            # 进入到这里已经是递归反向了
            last.next = head # last最先指向None, 而函数中的head最先指向倒数第二位，然后依次向前
            head.next = None # 最后一个head：val=为初始链表的val，next=None
            # 每次均会取出前一个head.val赋给pre
            return pre, head # 最终返回
        
        rt, m = Recur(head)
        return rt
```

例如例子中的题目，我们记录一下**递归反向**返回pre和last时`head`变量的取值：

![image-20220221205220422](C:\Users\Alex Qi\AppData\Roaming\Typora\typora-user-images\image-20220221205220422.png)

再来看一下最终返回时，head的取值

![image-20220221204405999](C:\Users\Alex Qi\AppData\Roaming\Typora\typora-user-images\image-20220221204405999.png)



而最终返回时pre的取值依次为：

![image-20220221204550158](C:\Users\Alex Qi\AppData\Roaming\Typora\typora-user-images\image-20220221204550158.png)

即链表通过last这个中间变量，每一次将当前head的个节点赋给新链表的next节点。注意直接的将head赋给last.next有风险，这样100%会形成环路，例如：`ListNode(val:5,next:ListNode(val:4,next:ListNode(val:5,next:None)))`

即相当于

`pre.next.next = head`

此时需要把head.next设置成None，才可以进行下一次递归。

### Useful algorithms&elements in Array 有用的算法和玩意儿
