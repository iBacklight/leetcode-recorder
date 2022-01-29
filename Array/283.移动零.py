#
# @lc app=leetcode.cn id=283 lang=python3
#
# [283] 移动零
#

# @lc code=start
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero_cnt = 0
        for n in range(len(nums)):
            if nums[n-zero_cnt] == 0:
                nums.pop(n-zero_cnt)
                nums.append(0)
                zero_cnt += 1
        # As solution, use double pointer
        # n = len(nums)
        # left = right = 0
        # while right < n:
        #     if nums[right] != 0:
        #         nums[left], nums[right] = nums[right], nums[left]
        #         left += 1
        #     right += 1
        # nums.sort(key=bool, reverse=True)
# @lc code=end

