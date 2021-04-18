import csv

from alive_progress import alive_bar

with open('fin.csv', 'r') as fin, open('fout.csv', 'w', newline='') as fout:

    # define reader and writer objects
    reader = csv.reader(fin, skipinitialspace=True)
    writer = csv.writer(fout, delimiter=',')
    # write headers
    writer.writerow(next(reader))

    # iterate and write rows based on condition
    for rows in reader:
        check = False
        for col in rows:
            if(str(col) == "Missing" or str(col) == "Unknown"):
                check = True        
        if(check == False):
            writer.writerow(rows)
                        