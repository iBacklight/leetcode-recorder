#
# @lc app=leetcode.cn id=565 lang=python3
#
# [565] 数组嵌套
#

# @lc code=start
from re import L


class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        # fast-slow pointer
        counter = max_num = temp = 0
        hash = dict()
        for slow in range(len(nums)):
            counter = 0
            temp = fast = slow
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

        # Split record index, and record 
        # dic = {}
        # l = len(nums)
        # tmp = [1 for _ in range(l)]
        # for i in range(l):
        #     if tmp[i]:
        #         j = i
        #         ct = 0
        #         while tmp[j]:
        #             tmp[j] = 0
        #             ct += 1
        #             j = nums[j]
        #         if j in dic:
        #             dic[i] = dic[j] + ct
        #         else:
        #             dic[i] = ct
        # return max(dic.values())
# @lc code=end

