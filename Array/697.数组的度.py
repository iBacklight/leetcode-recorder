#
# @lc app=leetcode.cn id=697 lang=python3
#
# [697] 数组的度
#

# @lc code=start
from cmath import inf


class Solution:
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
                 if (hash[num]['last_idx'] - hash[num]['first_idx']) < (hash[max_num]['last_idx'] - hash[max_num]['first_idx']):
                     max_num = num
            else:
                pass
        
        return (hash[max_num]['last_idx'] - hash[max_num]['first_idx'] + 1)
              
# @lc code=end

# [2,1,1,2,1,3,3,3,1,3,1,3,2]
