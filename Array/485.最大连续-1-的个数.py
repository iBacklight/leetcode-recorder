#
# @lc app=leetcode.cn id=485 lang=python3
#
# [485] 最大连续 1 的个数
#

# @lc code=start
class Solution:
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
        
        # cur = 0
        # max_num = 0
        # for num in nums:
        #     cur = (cur+num)*num
        #     if cur > max_num:
        #         max_num = cur 

        #     temp += n
        #     if temp
        return max_num
# @lc code=end

