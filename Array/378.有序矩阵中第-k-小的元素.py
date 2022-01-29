#
# @lc app=leetcode.cn id=378 lang=python3
#
# [378] 有序矩阵中第 K 小的元素
#

# @lc code=start
from cmath import inf
from statistics import mean, median, median_grouped


class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    #    return sorted(sum(matrix, []))[k-1]
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

        # bisect method
        # n = len(matrix)

        # def check(mid):
        #     i, j = n - 1, 0
        #     num = 0
        #     while i >= 0 and j < n:
        #         if matrix[i][j] <= mid:
        #             num += i + 1
        #             j += 1
        #         else:
        #             i -= 1
        #     return num >= k

        # left, right = matrix[0][0], matrix[-1][-1]
        # while left < right:
        #     mid = (left + right) // 2
        #     if check(mid):
        #         right = mid
        #     else:
        #         left = mid + 1
        
        # return left

            
# @lc code=end

