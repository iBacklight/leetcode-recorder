#
# @lc app=leetcode.cn id=21 lang=python3
#
# [21] 合并两个有序链表
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # violate backward
        if list1 == None: return list2
        if list2 == None: return list1
        newlist = cur = ListNode(0) 
        while True:
            if list2.val < list1. val:
                cur.next = list2
                if list2.next == None:
                    cur.next.next = list1
                    break
                else:
                    list2 = list2.next
            else:
                cur.next = list1
                if list1.next == None:
                    cur.next.next = list2
                    break
                else:
                    list1 = list1.next
            cur = cur.next
        return newlist.next 

        # violate forward
        if list1 == None: return list2
        if list2 == None: return list1
        newlist = cur = ListNode(0) 
        while True:
            if list2.val < list1. val:
                cur.next = list2
                list2 = list2.next
            else:
                cur.next = list1
                list1 = list1.next
            cur = cur.next
        cur.next = list1 if list1 else list2
        return newlist.next
         
# @lc code=end

