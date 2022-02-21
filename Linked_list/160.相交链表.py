#
# @lc app=leetcode.cn id=160 lang=python3
#
# [160] 相交链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # Hash table
        curA = headA
        curB = headB
        hash_midA = dict()
        hash_midB = dict()
        while (curA != None ) or (curB != None):
            if curA:
                hash_midA[curA] = 0
                if curA in hash_midB.keys():
                    return curA
                curA = curA.next
            if curB:
                hash_midB[curB] = 0 
                if curB in hash_midA.keys():
                    return curB
                curB = curB.next 
        return None

        # hash table 2
        # s = dict()
        # while headA:
        #     s[headA] = 0
        #     headA = headA.next
        # while headB:
        #     if headB in s.keys():
        #         return headB
        #     headB = headB.next
        # return None

        # Double pointer
        # if not headA or not headB:
        #     return None
        # nodeA = headA
        # nodeB = headB
        # while(nodeA !=nodeB):
        #     nodeA = nodeA.next if nodeA else headB
        #     nodeB = nodeB.next if nodeB else headA
        # return nodeA
        
        return None
# @lc code=end

