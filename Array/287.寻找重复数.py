#
# @lc app=leetcode.cn id=287 lang=python3
#
# [287] 寻找重复数
#

# @lc code=start
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        cnt = dict()
        for num in nums:
            if num in cnt.keys():
                return num
            else:
                cnt[num] = 1

        # Bisect Search
        left, right = 1, len(nums)-1
        cnt = 0
        while(left <= right):
            mid = (left+right)//2
            for num in nums:
                if  num <= mid:
                    cnt += 1
            if cnt <= mid:
                left = mid + 1
            else:
                right = mid-1
                dup = mid
            cnt = 0
        return dup

        # math method
        # return (sum(nums) - sum(set(nums))) // (len(nums) - len(set(nums)))

        # Fast-slow pointer
        # slow = fast = cir_start = 0
        # while True:
        #     fast = nums[nums[fast]]
        #     slow = nums[slow]
        #     if fast == slow: # utilize fast pointer find loop
        #         break

        # while True: 
        #     slow = nums[slow]
        #     cir_start = nums[cir_start]# set same pace for two pointer to find dup
        #     if cir_start == slow:
        #         return slow                         
# @lc code=end

