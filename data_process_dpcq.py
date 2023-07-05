

from novel_processor import NovelProcessor

class TXT2Json_processor(NovelProcessor):
    def createlines(self):
        with open(self.path, 'r') as f:
            ls = f.readlines()
        ls = self.clear_blank(ls)
        ls = self.clear_wrap(ls)
        return ls
    def createinputs(self):
        content_sep = []
        result_template = []
        is_narration = []
        for l in self.lines:
            l = l.replace('”', '"')
            l = l.replace('“', '"')
            if '"' not in l:
                is_narration.append(1)
            else:
                is_narration.append(0)
        for idx in range(len(self.lines)):
            cs = self.find_quoted_substrings(self.lines[idx])
            content_sep.append(cs)
            if len(cs) == 0:
                is_narration[idx] = ['旁白']
            else:
                is_narration[idx] = []
            result_template.append({
                'id': idx,
                'speaker': is_narration[idx],
                'content': self.lines[idx].replace('\n', '<n>'),
                "content_sep": content_sep[idx]
            })
        MAX_Length = 768

        instruction = "<n>根据以上内容回答问题。<n>"

        baseprompt = "这句话的说话人是谁？  [回答]"

        reserve_length = MAX_Length - len(instruction) - len(baseprompt)

        instruction_data = []
        source_idx = []

        for idx in range(len(result_template)):
            if result_template[idx]['speaker'] == 1:
                continue
            present_txt = result_template[idx]['content']
            for quote in result_template[idx]['content_sep']:
                left_win = (reserve_length - len(present_txt) - len(quote)) * 0.7
                right_win = (reserve_length - len(present_txt) - len(quote)) * (1 - 0.7)
                left_pt = idx - 1
                right_pt = idx + 1
                context_pre = ''
                while left_pt > 0 and len(result_template[left_pt]['content']) <= left_win:
                    context_pre = result_template[left_pt]['content'] + context_pre
                    left_win -= len(result_template[left_pt]['content'])
                    left_pt -= 1
                context_next = ''
                while right_pt < len(result_template) and len(result_template[right_pt]['content']) <= right_win:
                    context_pre = context_pre + result_template[right_pt]['content']
                    right_win -= len(result_template[right_pt]['content'])
                    right_pt += 1
                inputs = context_pre + present_txt + context_next + instruction + quote + baseprompt
                instruction_data.append(inputs)
                source_idx.append(idx)
        return instruction_data, result_template, source_idx
    def saveinputs(self):
        import json
        with open(self.output_path[0], 'w') as f:
            for line in self.inputs[0][:50]:
                f.write(line + '\n')
        with open(self.output_path[1], 'w') as f:
            json.dump(self.inputs[1], f)
        with open(self.output_path[2], 'w') as f:
            json.dump(self.inputs[2][:50], f)


def main():
    tps = TXT2Json_processor(path = '斗破苍穹.txt', output_path = ['./dpcq/test.source', './dpcq/result_template.json', './dpcq/source_idx.json'])

if __name__ == "__main__":
    main()




