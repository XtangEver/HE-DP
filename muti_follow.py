from dataclasses import asdict

class Multifollow():
    def __init__(self,query,history_record):
        self.query = query
        self.history_record = history_record

    def is_multi(self):
        """判断是否为多轮,以及返回第二轮的关键字"""
        multi_flag = False
        public_str = ""
        if self.history_record and self.history_record[-1].dialogue:
            print(f"------------历史记录---------------\n {[asdict(ch) for ch in self.history_record[-1].dialogue[0].get('user')]}")  # 查看对话历史
            dialogue_state = self.history_record[-1].dialogue[0].get('user')
            if "catalogue" in dialogue_state and dialogue_state["catalogue"] == "Multi_follow":
                # 取出上一轮AI回答的结果
                pre_ai_answer = self.history_record[-1].dialogue[1].get("user")
                if len(self.query > 4):  # 用户第二轮询问长度小于4的直接略过，不考虑
                    public_str,length = self.longest_public_str(pre_ai_answer,self.query)
                    if length:  # 如果存在匹配长度，则说明为多轮，则补充参数返回
                        print(f"----------------{public_str}-----------------")
                        multi_flag = True
        return multi_flag, public_str

    def reply_info(self,public_str):
        """根据用户询问返回参数信息"""
        dialogue_state = self.history_record[-1].dialogue[0].get('user')
        param = dialogue_state["Multi_follow"]["slot_values"]  # 返回一个字典
        param["details"] = public_str
        return param


    def longest_public_str(self,s1, s2):
        """最长公共子串匹配
        input:s1,s2
        output:最长公共子序列、长度
        """
        # 生成0矩阵，为方便后续计算，比字符串长度多了一列
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        mmax = 0  # 最长匹配的长度
        p = 0  # 最长匹配对应在s1中的最后一位
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - mmax:p], mmax  # 返回最长子串及其长度
