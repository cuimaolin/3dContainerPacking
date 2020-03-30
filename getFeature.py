import pandas as pd
import os
import re

def getFeaure(ln_root, ln_ult):
    # 利用pandas建立一个空表，每一列为一个特征
    data = pd.DataFrame(columns=("cnt_num", "cnt_species", "cnt_len_en", "cnt_wid_en", \
                                "ave_len", "ave_wid", "ave_high", "cnt_num/cnt_species",\
                                "ave_vol", "max_vol", "min_vol", "ave_len*wid*high",\
                                "cnt_vol", "utilization"))
    # 读取利用率所在的文件夹
    ult = pd.read_excel(ln_ult)
    # 得到data文件夹下的所有文件名
    filename = os.listdir(ln_root)
    # 得到data文件夹下文件的数量
    num = len(filename)
    # 遍历所有文件
    for i in range(0, num):
        # 得到当前txt的目录
        ln_data = os.path.join(ln_root, filename[i])
        # 读取当前txt所对应的利用率表
        set_name = filename[i][:-4]  # 去掉后缀.txt
        ult_cur = ult[ult['test set'] == set_name]
        # 打开当前txt
        file_data = open(ln_data, 'r')
        # 得到txt的所有行，将存到一个列表
        lines = file_data.readlines()
        # 关闭文件
        file_data.close()
        # 得到txt的行数
        num_line = len(lines)
        # 读取第四行的数据，即该txt每组数据有多少种箱型
        num_type = int(re.findall(r"\d+\.?\d*", lines[3])[0])
        # 从第二行开始循环，直到结束，每隔3+num_type一次(即一组数据)
        for j in range(1, num_line, 3 + num_type):
            idx = int(re.findall(r"\d+\.?\d*", lines[j])[0])
            # 得到利用率
            utilization = ult_cur[ult_cur[' test case'] == idx][' utilization'].tolist()[0]
            """
            以下为进行特征工程部分，是照着论文实现的(虽然很多特征看着很傻逼
            """
            cnt_num = 0
            cnt_species = num_type
            cnt_len_en = 0
            cnt_wid_en = 0
            ave_len = 0
            ave_wid = 0
            ave_high = 0
            max_vol = 0
            cnt_vol = 0
            min_vol = float('inf')
            for k in range(j+3, j+3+num_type):
                line = re.findall(r"\d+\.?\d*", lines[k])
                line = list(map(int, line))
                cnt_num += line[7]
                cnt_len_en += line[2] * line[7]
                cnt_wid_en += line[4] * line[7]
                ave_len += line[1] * line[7]
                ave_wid += line[3] * line[7]
                ave_high += line[5] * line[7]
                vol = line[1] * line[3] * line[5]
                if vol > max_vol:
                    max_vol = vol
                if vol < min_vol:
                    min_vol = vol
                cnt_vol += vol * line[7]
            ave_len /= cnt_num
            ave_wid /= cnt_num
            ave_high /= cnt_num
            ave_vol = cnt_vol / cnt_num
            # 将求得到特征添加到表中
            data = data.append({"cnt_num":cnt_num, "cnt_species":cnt_species,\
                                "cnt_len_en":cnt_len_en, "cnt_wid_en":cnt_wid_en, \
                                "ave_len":ave_len, "ave_wid":ave_wid, "ave_high":ave_high,\
                                "cnt_num/cnt_species":cnt_num/cnt_species, "ave_vol":ave_vol,\
                                "max_vol":max_vol, "min_vol":min_vol, 
                                "ave_len*wid*high":ave_len*ave_wid*ave_high,\
                                "cnt_vol":cnt_vol, "utilization":utilization}, ignore_index=True)
    return data