import csv
import numpy as np

def given_column_data(project_path, file_name_and_file_path, data, firstrow):
    with open(project_path + file_name_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(firstrow)
        for i in range(0, len(data[0])):
            row = []
            for j in range(0, len(data)):
                row.append(data[j][i])
            spamwriter.writerow(row)


def given_column_data1(project_path, file_name_and_file_path, data, first_row, first_column):
    with open(project_path + file_name_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(first_row)
        for i in range(0, len(data[0])):
            row = []
            row.append(first_column[i])
            for j in range(0, len(data)):
                row.append(data[j][i])
            spamwriter.writerow(row)

def write_matrix(project_path, file_name_and_file_path, data, first_row, first_row_or_not=False):
    with open(project_path + file_name_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if first_row_or_not is True:
            spamwriter.writerow(first_row)
        for i in range(0, len(data)):
            row = []
            for j in range(0, len(data[i])):
                row.append(data[i][j])
            spamwriter.writerow(row)

def given_column_data_without_title(project_path, file_name_and_file_path, data):
    with open(project_path + file_name_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(data[0])):
            row = []
            for j in range(0, len(data)):
                row.append(data[j][i])
            spamwriter.writerow(row)

def given_column_data2(project_path, file_name_and_file_path, data, first_row, first_column):
    with open(project_path + file_name_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        d = [""]
        for i in first_row:
            d.append(i)
        spamwriter.writerow(d)
        for i in range(0, len(data[0])):
            row = []
            row.append(first_column[i])
            for j in range(0, len(data)):
                row.append(data[j][i])
            spamwriter.writerow(row)

def write_predicted_data(project_path, file_path_and_file_path, data):
    with open(project_path + file_path_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(data)):
            row = []
            for j in range(0, len(data[i])):
                row.append(data[i][j])
            spamwriter.writerow(row)

def given_row_data(project_path, file_path_and_file_path, data, first_row, timeinfo):
    with open(project_path + file_path_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(first_row)
        for i in range(0, len(data)):
            row = [timeinfo[i]]
            for j in range(0, len(data[i])):
                row.append(data[i][j])
            spamwriter.writerow(row)


def write_clustering_gps_data(project_path, file_name_and_file_path, location, time, clustering_index, userid, datatype):
    with open(project_path + file_name_and_file_path + '.csv', 'w', newline='', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # row = ["Lon", "Lat", "Time", "Cluster Index"]
        # spamwriter.writerow(row)
        for i in range(0, len(location)):
            row = []
            row.append(location[i][0])
            row.append(location[i][1])
            row.append(time[i])
            row.append(clustering_index[i])
            row.append(userid[i])
            row.append(datatype[i])
            spamwriter.writerow(row)