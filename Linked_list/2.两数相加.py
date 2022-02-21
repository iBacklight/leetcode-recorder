#
# @lc app=leetcode.cn id=2 lang=python3
#
# [2] 两数相加
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # num1 = 0
        # num2 = 0
        # cur_node = l1
        # i = 0
        # while cur_node:
        #      num1 += cur_node.val*pow(10,i)
        #      cur_node = cur_node.next
        #      i += 1
        # cur_node = l2
        # i = 0
        # while cur_node:
        #      num2 += cur_node.val*pow(10,i)
        #      cur_node = cur_node.next
        #      i += 1
        # new_num = num1 + num2
        # new_node = ListNode()
        # cur_node = new_node
        # while True:
        #     cur_node.val = new_num % 10
        #     new_num = new_num//10
        #     if new_num != 0:
        #         cur_node.next = ListNode()
        #         cur_node = cur_node.next
        #     else:
        #         break
        # return new_node
        
        newPoint = ListNode(l1.val + l2.val)
        rt, tp = newPoint, newPoint
        while (l1 and (l1.next != None)) or (l2 and (l2.next != None)) or (tp.val > 9):
            l1, l2 = l1.next if l1 else l1, l2.next if l2 else l2
            tmpsum = (l1.val if l1 else 0) + (l2.val if l2 else 0)
            tp.next = ListNode(tp.val//10 + tmpsum)
            tp.val %= 10
            tp = tp.next
        return rt


# @lc code=end

