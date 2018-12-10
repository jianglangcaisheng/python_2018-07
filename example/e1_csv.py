
import csv

# 功能：将一个二重列表写入到csv文件中
# 输入：文件名称，数据列表
def createListCSV(fileName="", dataList=[]):
    with open(fileName, "wb") as csvFile:
        csvWriter = csv.writer(csvFile)
        for data in dataList:
            csvWriter.writerow(data.to_bytes())
        csvFile.close

createListCSV(r"F:\0 SoG\0 SOG_201807\0 SOG_201807\0_history_python/123.csv", [[1, 2], [3, 4]])
