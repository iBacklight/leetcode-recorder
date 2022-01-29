#
# @lc app=leetcode.cn id=566 lang=python3
#
# [566] 重塑矩阵
#

# @lc code=start
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        cr = len(mat)
        cc = len(mat[0])
        reshaped_mat = []
        i = j = 0
        if cr*cc != r*c:
            return mat
        else:
            for row in range(r):
                reshaped_mat.append([])
                for col in range(c):
                    if j >= cc:
                        j = 0
                        i += 1
                    reshaped_mat[row].append(mat[i][j])
                    j += 1
        return reshaped_mat
# @lc code=end

