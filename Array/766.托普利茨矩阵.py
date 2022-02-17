#
# @lc app=leetcode.cn id=766 lang=python3
#
# [766] 托普利茨矩阵
#

# @lc code=start
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        for row in range(len(matrix)-1):
            for col in range(len(matrix[0])-1):
                if matrix[row][col] != matrix[row+1][col+1]:
                    return False
        # One line version
        # return all(matrix[i][:-1] == matrix[i+1][1:] for i in range(len(matrix)-1))
        return True

        # Eight queen
        # R, C = len(matrix), len(matrix[0])
        # rc_diff = defaultdict(int)     
        # for r in range(R):
        #     for c in range(C):
        #         if r-c not in rc_diff:
        #             rc_diff[r-c] = matrix[r][c]
        #         else:
        #             if rc_diff[r-c] != matrix[r][c]:
        #                 return False
        # return True

# @lc code=end

