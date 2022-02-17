#
# @lc app=leetcode.cn id=769 lang=python3
#
# [769] 最多能完成排序的块
#

# @lc code=start
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        cnt = 0
        cur_max = -1
        n = len(arr)
        for idx in range(n):       
            cur_max = max(cur_max, arr[idx])
            if cur_max == idx:
                cnt += 1
        return cnt
# @lc code=end

