import re
import abc
import string

class NovelProcessor(metaclass=abc.ABCMeta):
    def __init__(self, path, output_path):
        self.path = path
        self.output_path = output_path
        self.lines = self.createlines()
        self.inputs = self.createinputs()
        self.saveinputs()
    @abc.abstractmethod
    def createlines(self):
        pass

    @abc.abstractmethod
    def createinputs(self):
        pass

    @abc.abstractmethod
    def saveinputs(self):
        pass


    def is_english_punctuation(self, char):
        if char in string.punctuation:
            return True
        else:
            return False

    # 判断一个字符是否为中文标点符号
    def is_chinese_punctuation(self, char):
        if re.match(r'[\u3000-\u303F\uFF00-\uFFEF]', char):
            return True
        else:
            return False

    # 判断一个字符是否为中英文标点符号
    def is_punctuation(self, char):
        if self.is_english_punctuation(char) or self.is_chinese_punctuation(char):
            return True
        else:
            return False

    def find_quoted_substrings(self, s):
        l = s.replace('”', '"')
        l = l.replace('“', '"')
        l = l.replace('…', '。')
        pattern = r'"(.*?)"'
        substrings = re.findall(pattern, l)
        result = []
        for str in substrings:
            if len(str) > 0 and self.is_punctuation(str[-1]):
                ##以后加上针对中英文不同的处理

                result.append('“' + str + '”')
        return result

    def clear_blank(self,lines):
        result = []
        for l in lines:
            if len(l.replace(' ', '')) > 0:
                result.append(l.replace(' ', ''))
        return result

    def clear_wrap(self,lines):
        result = []
        for l in lines:
            if len(l.replace('\n', '')) > 0:
                result.append(l.replace(' ', ''))
        return result