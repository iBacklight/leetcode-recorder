#
# @lc app=leetcode.cn id=645 lang=python3
#
# [645] 错误的集合
#

# @lc code=start
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        # violate force method
        # n = len(nums)
        # temp = []
        # wrongvalue = -1
        # rightvalue = -1
        # for i in range(n):
        #         try:
        #             nums.index(i+1)
        #             if nums[i] in temp:
        #                 wrongvalue = nums[i]
        #                 if rightvalue != -1:
        #                     break
        #             else:
        #                 temp.append(nums[i])
        #         except:
        #         rightvalue = i+1
        #         if nums[i] in temp:
        #                 wrongvalue = nums[i]
        #                 break
        #         else:
        #                 temp.append(nums[i])
        # return ([wrongvalue, rightvalue])

        # HashTable
        # errvalue = -1
        # rightvalue = -1
        # cnt = dict()
        # for key in range(1,len(nums)+1):
        #     cnt[key] = 0
        # for num in nums:
        #     cnt[num] += 1
        #     if cnt[num] > 1:
        #         errvalue = num
        # for key in range(1,len(nums)+1):
        #     if cnt[key] == 0:
        #         rightvalue = key
        #         break
        # return [errvalue, rightvalue]

        # Counter 
        from collections import Counter
        check = Counter(nums)
        dup = check.most_common(1)[0][0]
        for i in range(1, len(nums) + 1):
            if check[i] == 0:
                abs = i
                break
            # abs = [i for i in range(1, len(nums) + 1) if check[i] == 0][0]
        return [dup, abs]

        # math
        temp = sum(set(nums))
        return [sum(nums)-temp, sum(range(1,len(nums)+1)) - temp]
# @lc code=end

