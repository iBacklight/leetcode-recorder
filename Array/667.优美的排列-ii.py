#
# @lc app=leetcode.cn id=667 lang=python3
#
# [667] 优美的排列 II
#

# @lc code=start
class Solution:
    def constructArray(self, n: int, k: int) -> List[int]:
        # marked as non-solved
        res = list(range(1, n + 1))  # At first there is a different difference in absolute value 
        for i in range(1, k):  # Each flip behind produces a new
            res[i:] = res[:i-1:-1]  # flip
        return res
       
       # official solution: Construction
        result = [1]
        flag = True
        for i in range(k, 0, -1):
            if flag:
                result.append(result[-1] + i)
                flag = False
            else:
                result.append(result[-1] - i)
                flag = True
        for i in range(1 + k + 1, n + 1):
            result.append(i)
        return result

# @lc code=end

