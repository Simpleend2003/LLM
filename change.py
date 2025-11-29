import csv


def tsv_to_csv(tsv_file, csv_file):
    # 打开 TSV 文件
    with open(tsv_file, 'r', newline='', encoding='utf-8') as tsv_f:
        # 使用 csv.reader 读取 TSV，指定 delimiter 为 '\t'
        tsv_reader = csv.reader(tsv_f, delimiter='\t')

        # 打开 CSV 文件进行写入
        with open(csv_file, 'w', newline='', encoding='utf-8') as csv_f:
            csv_writer = csv.writer(csv_f)

            # 将 TSV 内容写入 CSV
            for row in tsv_reader:
                csv_writer.writerow(row)

    print(f"转换完成！{tsv_file} 已转换为 {csv_file}")



# 调用函数，提供输入 TSV 文件和输出 CSV 文件的路径
tsv_to_csv('data/tram_train.tsv', 'data/tram_train.csv')
