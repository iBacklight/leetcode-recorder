#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # # violate poor solution
        # for id1,num1 in enumerate(nums):
        #     for id2,num2 in enumerate(nums):
        #         if id1 == id2:
        #             continue
        #         if num1 + num2 == target:
        #             return [id1, id2]
        # Hash Table
        hashmap = {}
        for id,num in enumerate(nums):
            gap = target - num
            if gap in hashmap:
                return [id, hashmap[gap]]
            else:
                hashmap[num] = id

# @lc code=end

