#
# @lc app=leetcode.cn id=206 lang=python3
#
# [206] 反转链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # if not head:
        #     return None
        # l = 0
        # hash_mid = dict()
        # while head != None:
        #     hash_mid[l] = head.val
        #     l += 1
        #     head = head.next
        # new_node = temp = ListNode(hash_mid[l-1])
        # for n in reversed(range(l-1)):
        #     temp.next = ListNode(hash_mid[n])
        #     temp = temp.next
        # return new_node 

        # # Simple version of iteration
        # if not head:
        #      return None
        # prev = None
        # curr = head
        # while curr != None:
        #     temp = curr.next
        #     curr.next = prev
        #     prev = curr
        #     curr = temp
        # return prev
            
        # Recursion
        def Recur(head):
            if head == None or head.next == None:# head == None avoids input == []
                return head, head
            pre, last = Recur(head.next)
            last.next = head
            head.next = None
            return pre, head
        
        rt, m = Recur(head)
        return rt      
# @lc code=end

