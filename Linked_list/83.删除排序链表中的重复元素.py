#
# @lc app=leetcode.cn id=83 lang=python3
#
# [83] 删除排序链表中的重复元素
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head is None:
            return head
        ht = dict()
        ht[head.val] = 1
        cur = head
        while(cur.next is not None):
            if cur.next.val in ht.keys():
               cur.next = cur.next.next if cur.next.next else None
               if cur.next == None:
                   break
            else:
                ht[cur.next.val] = 1
                cur = cur.next
        return head

        # if not head:
        #     return head

        # cur = head
        # while cur.next:
        #     if cur.val == cur.next.val:
        #         cur.next = cur.next.next
        #     else:
        #         cur = cur.next

        # return head

# @lc code=end

